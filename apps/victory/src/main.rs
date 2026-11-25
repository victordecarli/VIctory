//! `victory`: the command-line front end.
//!
//! Every command reads the user's own files and writes only to stdout or to
//! the user's cache directory (outside the repository).
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "a command-line tool: its reports are its output"
)]

mod census;
mod inspect;
mod run;
#[cfg(windows)]
mod run_log;
mod shader;
mod translate;
mod walk;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "victory", version, about = "Run an Xbox Series title natively on Windows")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Summarise one PE image: headers, sections, imports, TLS, build id.
    Inspect {
        file: PathBuf,
        /// List every imported function.
        #[arg(long)]
        imports: bool,
        /// List every exported function.
        #[arg(long)]
        exports: bool,
    },
    /// Aggregate the imports of every PE image under the given paths and
    /// show which ones our runtime covers. Defaults to the configured
    /// game and test-title directories.
    Census {
        paths: Vec<PathBuf>,
        /// Do not count host pass-through as covered.
        #[arg(long)]
        strict: bool,
        /// List every missing `dll!function`, most used first.
        #[arg(long)]
        missing: bool,
    },
    /// List the `dll!export` pairs an import library describes.
    LibExports {
        lib: PathBuf,
        /// List every export, not just per-DLL counts.
        #[arg(long)]
        list: bool,
    },
    /// Find shader containers in files or directories (also inside engine
    /// archives, if uncompressed) and report their layout.
    ShaderInfo {
        paths: Vec<PathBuf>,
        /// Print one line per container found.
        #[arg(long)]
        verbose: bool,
        /// Also list the decoded native instructions (implies --verbose).
        #[arg(long)]
        disasm: bool,
    },
    /// Run every shader found (containers, SGD5 files) through the
    /// translator and report what it cannot handle yet.
    Translate {
        paths: Vec<PathBuf>,
        /// Check emitted DXIL with dxil.dll's validator (`VICTORY_DXIL`,
        /// else the newest Windows SDK's).
        #[arg(long)]
        validate: bool,
        /// Also create a host D3D12 pipeline for each emitted shader.
        #[arg(long)]
        pipelines: bool,
        /// Print one line per program and outcome.
        #[arg(long)]
        list: bool,
        /// Worker threads (default: every logical core).
        #[arg(long)]
        jobs: Option<usize>,
        /// Translate for a host whose waves have at most this many lanes
        /// (32 on NVIDIA): console wave64 programs that can run as two
        /// waves of 32 do. The default, 64, splits nothing.
        #[arg(long, default_value_t = victory_shader::emit::CONSOLE_WAVE)]
        host_lanes: u32,
        /// Also write the counts per kind of program as JSON to this file
        /// (read by `cargo xtask progress --record`). Needs --validate: a
        /// shader counts as translated once the validator accepts it.
        #[arg(long, requires = "validate")]
        json: Option<PathBuf>,
    },
    /// Prepare, map and bind a title executable; with --execute, run it.
    Run {
        /// The title's executable. Default: the only `.exe` at the top of
        /// the configured game_dir.
        exe: Option<PathBuf>,
        /// What unimplemented imports do when called.
        #[arg(long, value_enum, default_value_t = Unimplemented::Trap)]
        unimplemented: Unimplemented,
        /// Disable host pass-through (everything must be implemented by us).
        #[arg(long)]
        strict: bool,
        /// Actually start the title. Without it, stop after binding.
        #[arg(long)]
        execute: bool,
        /// The console model the title sees (device type, memory).
        #[arg(long, value_enum, default_value_t = Console::SeriesX)]
        console: Console,
        /// Arguments for the title, after `--` (it reads the whole command
        /// line; a title that relaunches itself passes some).
        #[arg(last = true)]
        title_args: Vec<String>,
    },
    /// Show the resolved configuration and check it.
    Config,
}

#[derive(Clone, Copy, ValueEnum)]
enum Unimplemented {
    Trap,
    Zero,
}

#[derive(Clone, Copy, ValueEnum)]
enum Console {
    SeriesX,
    SeriesS,
}

fn main() -> Result<()> {
    // The audio recorder `run` starts for `VICTORY_AUDIO_CAPTURE`: a
    // process of its own, with a log file of its own.
    #[cfg(all(windows, feature = "diagnostics"))]
    if let Some(parent) = victory_hle::modules::audio::capture::parent() {
        start_run_log();
        run_log::start("audio-capture", None, Some(parent));
        victory_hle::modules::audio::capture::child();
        return Ok(());
    }
    let cli = Cli::parse();
    if matches!(cli.command, Command::Run { .. }) {
        start_run_log();
    } else {
        victory_core::logging::init("info");
    }
    match cli.command {
        Command::Inspect { file, imports, exports } => inspect::run(&file, imports, exports),
        Command::Census { paths, strict, missing } => census::run(paths, strict, missing),
        Command::LibExports { lib, list } => inspect::lib_exports(&lib, list),
        Command::ShaderInfo { paths, verbose, disasm } => {
            shader::run(paths, verbose || disasm, disasm)
        }
        Command::Translate { paths, validate, pipelines, list, jobs, host_lanes, json } => {
            translate::run(paths, validate, pipelines, list, jobs, host_lanes, json)
        }
        Command::Run { exe, unimplemented, strict, execute, console, title_args: _ } => {
            run::run(&title_exe(exe)?, unimplemented, strict, execute, console)
        }
        Command::Config => show_config(),
    }
}

/// Logs to the console and to a run log file in the log directory; to the
/// console only if there is no usable log directory (the command then
/// fails on the configuration too, and says why).
fn start_run_log() {
    match victory_core::Config::load().and_then(|cfg| cfg.prepared_log_dir()) {
        Ok(dir) => {
            victory_core::logging::init_run("info", &dir);
        }
        Err(e) => {
            victory_core::logging::init("info");
            tracing::warn!(error = format!("{e:#}"), "no run log file: no log directory");
        }
    }
}

fn show_config() -> Result<()> {
    let cfg = victory_core::Config::load()?;
    let status = |p: &Option<PathBuf>| match p {
        Some(p) if p.exists() => format!("{}", p.display()),
        Some(p) => format!("{}  (does not exist)", p.display()),
        None => "(not set)".to_owned(),
    };
    println!(
        "config file : {}",
        cfg.source.as_ref().map_or("(none)".into(), |p| p.display().to_string())
    );
    println!("game_dir    : {}", status(&cfg.game_dir));
    println!("cache_dir   : {}", cfg.cache_dir.display());
    println!("data_dir    : {}", cfg.data_dir.display());
    println!("log_dir     : {}", cfg.log_dir.display());
    for t in &cfg.test_titles {
        println!("test_title  : {}", status(&Some(t.clone())));
    }
    let derived = [&cfg.cache_dir, &cfg.data_dir, &cfg.log_dir];
    for p in cfg.game_dir.iter().chain(&cfg.test_titles).chain(derived) {
        if let Err(e) = victory_core::ensure_outside_repo(p) {
            println!("POLICY VIOLATION: {e}");
        }
    }
    Ok(())
}

/// Paths given on the command line, or the configured game + test titles.
pub(crate) fn default_paths(paths: Vec<PathBuf>) -> Result<Vec<PathBuf>> {
    if !paths.is_empty() {
        return Ok(paths);
    }
    let cfg = victory_core::Config::load()?;
    let all: Vec<PathBuf> = cfg.game_dir.into_iter().chain(cfg.test_titles).collect();
    anyhow::ensure!(
        !all.is_empty(),
        "no paths given and none configured (see victory.example.toml)"
    );
    Ok(all)
}

/// `exe`, or the configured game's: the only `.exe` at the top of
/// `game_dir`.
fn title_exe(exe: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(exe) = exe {
        return Ok(exe);
    }
    let cfg = victory_core::Config::load()?;
    let dir = cfg
        .game_dir
        .context("no executable given and no game_dir configured (see victory.example.toml)")?;
    only_exe(&dir)
}

/// The only `.exe` file directly in `dir`.
fn only_exe(dir: &Path) -> Result<PathBuf> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry.with_context(|| format!("reading {}", dir.display()))?.path();
        if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("exe")) && path.is_file() {
            found.push(path);
        }
    }
    found.sort();
    match found.as_slice() {
        [one] => Ok(one.clone()),
        [] => anyhow::bail!("no .exe in {}; give the title's executable", dir.display()),
        several => {
            let names: Vec<String> = several.iter().map(|p| p.display().to_string()).collect();
            anyhow::bail!(
                "several executables in {}: {}; give the one to run",
                dir.display(),
                names.join(", ")
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_takes_the_executable_or_nothing() {
        assert!(Cli::try_parse_from(["victory", "run", "--execute"]).is_ok());
        assert!(Cli::try_parse_from(["victory", "run", "title.exe", "--execute"]).is_ok());
    }

    #[test]
    fn the_only_exe_of_a_folder_is_the_title() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("title.exe"), b"").unwrap();
        std::fs::write(dir.path().join("library.dll"), b"").unwrap();
        std::fs::create_dir(dir.path().join("tools.exe")).unwrap();
        assert_eq!(only_exe(dir.path()).unwrap(), dir.path().join("title.exe"));
    }

    #[test]
    fn a_folder_with_no_exe_or_several_names_none() {
        let dir = tempfile::tempdir().unwrap();
        let none = only_exe(dir.path()).unwrap_err().to_string();
        assert!(none.contains("no .exe"), "{none}");
        std::fs::write(dir.path().join("a.exe"), b"").unwrap();
        std::fs::write(dir.path().join("b.EXE"), b"").unwrap();
        let several = only_exe(dir.path()).unwrap_err().to_string();
        assert!(several.contains("a.exe") && several.contains("b.EXE"), "{several}");
    }

    #[test]
    fn translate_json_needs_validate() {
        assert!(Cli::try_parse_from(["victory", "translate", "--json", "s.json"]).is_err());
        assert!(
            Cli::try_parse_from(["victory", "translate", "--validate", "--json", "s.json"]).is_ok()
        );
    }
}

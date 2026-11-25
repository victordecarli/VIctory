//! One tracing setup for every binary.
//!
//! Filter with `VICTORY_LOG` (falls back to `RUST_LOG`), e.g.
//! `VICTORY_LOG=victory_loader=trace,info`. Debug and trace output exist
//! only in builds with the `diagnostics` feature; a plain release build
//! logs info and above whatever the filter asks for. A filter that does not
//! parse is replaced by the default and reported once logging is up.
//!
//! [`init`] logs to the console only. [`init_run`] also writes the run's
//! log file, `run-<UTC time>-<pid>.jsonl` (one per process, the newest
//! [`KEEP_RUNS`] kept): one JSON object per line, the event's fields beside
//! `timestamp` (UTC), `level`, `target`, `threadId` and `threadName`, so a
//! field must not be named after one of those. Each event is written when
//! it happens, unbuffered, so a crash loses nothing logged before it.
//!
//! Events with the [`SUMMARY`] target pass whatever the filter says, on the
//! console and in the file: the run's start, its periodic reports, the
//! host adapter, a crash ([`append_crash`]).

mod run_file;
#[cfg(test)]
mod tests;

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use tracing::Subscriber;
use tracing_subscriber::filter::{FilterExt, LevelFilter, filter_fn};
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::layer::{Filter, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

pub use run_file::{KEEP_RUNS, append_crash};

/// The target of events that are always logged, whatever the filter.
pub const SUMMARY: &str = "victory::summary";

/// The most verbose level this build can log.
const CAP: LevelFilter =
    if cfg!(feature = "diagnostics") { LevelFilter::TRACE } else { LevelFilter::INFO };

/// The filter asked for, checked: what is applied, and why the request was
/// refused if it was.
pub struct Spec {
    directives: String,
    refused: Option<String>,
}

impl Spec {
    /// `VICTORY_LOG`, else `RUST_LOG`, else `default_level`.
    pub fn from_env(default_level: &str) -> Self {
        let read = |name: &str| match std::env::var(name) {
            Ok(text) => Some(Ok(text)),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(raw)) => {
                Some(Err(format!("{name} is not Unicode: {raw:?}")))
            }
        };
        Self::new(read("VICTORY_LOG").or_else(|| read("RUST_LOG")), default_level)
    }

    /// `asked` (or why it could not be read), else `default_level`.
    pub fn new(asked: Option<Result<String, String>>, default_level: &str) -> Self {
        let refused = match asked {
            None => None,
            Some(Ok(text)) => match EnvFilter::try_new(&text) {
                Ok(_) => return Spec { directives: text, refused: None },
                Err(e) => Some(format!("{text}: {e}")),
            },
            Some(Err(why)) => Some(why),
        };
        Spec { directives: default_level.to_owned(), refused }
    }

    /// Why the asked filter was not used, if it was not.
    pub fn refused(&self) -> Option<&str> {
        self.refused.as_deref()
    }

    /// The filter of one destination: the directives, or the summary target.
    fn filter<S>(&self) -> impl Filter<S> + Send + Sync + 'static
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        filter_fn(|meta| meta.target() == SUMMARY).or(EnvFilter::new(&self.directives))
    }
}

/// The subscriber for `spec`: compact text to `console` and, if given, a
/// JSON line per event to `file`. Built, not installed, so tests can give
/// it destinations in memory.
pub fn subscriber<C, F>(spec: &Spec, console: C, file: Option<F>) -> impl Subscriber + Send + Sync
where
    C: for<'w> MakeWriter<'w> + Send + Sync + 'static,
    F: for<'w> MakeWriter<'w> + Send + Sync + 'static,
{
    let console = tracing_subscriber::fmt::layer()
        .compact()
        .with_target(true)
        .with_writer(console)
        .with_filter(spec.filter());
    let file = file.map(|file| {
        tracing_subscriber::fmt::layer()
            .json()
            .flatten_event(true)
            .with_current_span(false)
            .with_span_list(false)
            .with_thread_ids(true)
            .with_thread_names(true)
            .with_ansi(false)
            .with_writer(file)
            .with_filter(spec.filter())
    });
    tracing_subscriber::registry().with(CAP).with(console).with(file)
}

/// Logs to the console only.
pub fn init(default_level: &str) {
    let spec = Spec::from_env(default_level);
    install(&spec, None);
    report_refused(&spec);
}

/// Logs to the console and to a new run file in `dir`, after removing all
/// but the newest [`KEEP_RUNS`] run files there. The file's path is the
/// first line logged; `None` (and a warning) if it could not be created,
/// leaving the console only.
pub fn init_run(default_level: &str, dir: &Path) -> Option<PathBuf> {
    let spec = Spec::from_env(default_level);
    let opened = run_file::open(dir, &run_file::now_stamp(), std::process::id());
    let (path, file) = match opened {
        Ok(opened) => opened,
        Err(e) => {
            install(&spec, None);
            tracing::warn!(dir = %dir.display(), error = %e, "no run log file; the console only");
            report_refused(&spec);
            return None;
        }
    };
    let pruned = run_file::prune(dir, KEEP_RUNS);
    install(&spec, Some(file));
    run_file::remember(path.clone());
    tracing::info!(target: SUMMARY, path = %path.display(), "run log");
    report_refused(&spec);
    for old in &pruned.removed {
        tracing::info!(path = %old.display(), "old run log removed");
    }
    for (old, e) in &pruned.failed {
        tracing::warn!(path = %old.display(), error = %e, "old run log not removed");
    }
    Some(path)
}

/// The `VICTORY_*` and `RUST_LOG` variables of `vars`, by name, for the
/// record of what a run was started with.
pub fn switches(vars: impl Iterator<Item = (OsString, OsString)>) -> Vec<(String, String)> {
    let mut kept: Vec<(String, String)> = vars
        .map(|(name, value)| {
            (name.to_string_lossy().into_owned(), value.to_string_lossy().into_owned())
        })
        .filter(|(name, _)| name.starts_with("VICTORY_") || name == "RUST_LOG")
        .collect();
    kept.sort();
    kept
}

fn install(spec: &Spec, file: Option<std::fs::File>) {
    if let Err(e) = subscriber(spec, std::io::stderr, file).try_init() {
        not_installed(&e);
    }
}

#[allow(clippy::print_stderr, reason = "no subscriber is installed to take the error")]
fn not_installed(e: &dyn std::fmt::Display) {
    eprintln!("VIctory: logging not set up: {e}");
}

fn report_refused(spec: &Spec) {
    if let Some(why) = spec.refused() {
        tracing::warn!(filter = %spec.directives, why, "log filter not understood; using the default");
    }
}

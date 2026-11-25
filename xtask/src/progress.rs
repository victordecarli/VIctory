//! `cargo xtask progress`: the README's shader progress block, generated
//! from a measurement so that its numbers come from a command, not from
//! numbers typed by hand.
//!
//! `victory translate --json <file>` measures (on Windows, over the user's
//! own game files); `--record <file>` stores that summary with the commit
//! it was built from in [`MEASUREMENT`]. Every run then rewrites the block
//! between [`START`] and [`END`] in `README.md`; `--check`, and the
//! `readme_progress_matches_the_measurement` test, fail when it differs.
//! The measurement holds aggregate counts only (`docs/asset-policy.md`).

use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use crate::audit::git;

/// The measurement, relative to the repository root.
pub const MEASUREMENT: &str = "docs/progress/shaders.json";
const START: &str = "<!-- progress:start -->";
const END: &str = "<!-- progress:end -->";
/// The `victory translate --json` format this reads.
const SCHEMA: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Kind {
    found: usize,
    emitted: usize,
}

/// What `victory translate --json` writes.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Summary {
    schema: u32,
    complete: bool,
    host_lanes: u32,
    vertex_pixel: Kind,
    compute: Kind,
    compute_wave64: usize,
    mesh: Kind,
}

/// A [`Summary`] and the commit the measuring build came from.
#[derive(Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Measurement {
    schema: u32,
    complete: bool,
    /// The widest wave the translation assumed (`--host-lanes`).
    host_lanes: u32,
    vertex_pixel: Kind,
    compute: Kind,
    /// Of `compute.emitted`, those emitted for 64-lane waves: not counted
    /// when `host_lanes` is below 64, since such a host refuses them.
    compute_wave64: usize,
    mesh: Kind,
    /// `git rev-parse --short=10 HEAD` when recorded.
    commit: String,
    /// Its committer date (`git show -s --format=%cs`).
    commit_date: String,
}

pub fn run(root: &Path, record: Option<&Path>, check: bool) -> Result<()> {
    let measurement = root.join(MEASUREMENT);
    if let Some(summary) = record {
        let text = std::fs::read_to_string(summary)
            .with_context(|| format!("reading {}", summary.display()))?;
        let commit = git(root, &["rev-parse", "--short=10", "HEAD"])?.trim().to_owned();
        let date = git(root, &["show", "-s", "--format=%cs", "HEAD"])?.trim().to_owned();
        let m = measured(&text, commit, date)?;
        std::fs::write(&measurement, serde_json::to_string_pretty(&m)? + "\n")
            .with_context(|| format!("writing {MEASUREMENT}"))?;
        println!("recorded {MEASUREMENT} at {}", m.commit);
    }
    let m: Measurement = serde_json::from_str(
        &std::fs::read_to_string(&measurement).with_context(|| format!("reading {MEASUREMENT}"))?,
    )
    .with_context(|| format!("parsing {MEASUREMENT}"))?;
    let path = root.join("README.md");
    let readme = std::fs::read_to_string(&path).context("reading README.md")?;
    let updated = splice(&readme, &render(&m))?;
    if check {
        ensure!(
            updated == readme,
            "README.md's progress block differs from {MEASUREMENT}: run `cargo xtask progress`"
        );
    } else if updated != readme {
        std::fs::write(&path, updated).context("writing README.md")?;
        println!("README.md progress block updated");
    }
    Ok(())
}

fn measured(summary: &str, commit: String, commit_date: String) -> Result<Measurement> {
    let s: Summary = serde_json::from_str(summary).context("parsing the translate summary")?;
    let Summary { schema, complete, host_lanes, vertex_pixel, compute, compute_wave64, mesh } = s;
    ensure!(
        schema == SCHEMA,
        "translate summary schema {schema}, this reads {SCHEMA}: update xtask/src/progress.rs"
    );
    Ok(Measurement {
        schema,
        complete,
        host_lanes,
        vertex_pixel,
        compute,
        compute_wave64,
        mesh,
        commit,
        commit_date,
    })
}

/// The share emitted in tenths of a percent, rounded down so that a kind
/// short of every program never shows 100.0%.
fn permille(k: Kind) -> Option<usize> {
    (k.found > 0).then(|| k.emitted.min(k.found) * 1000 / k.found)
}

fn percent(k: Kind) -> String {
    permille(k).map_or("–".into(), |p| format!("{}.{}%", p / 10, p % 10))
}

fn bar(k: Kind) -> String {
    const CELLS: usize = 20;
    let full = permille(k).map_or(0, |p| p * CELLS / 1000);
    format!("`{}{}`", "█".repeat(full), "░".repeat(CELLS - full))
}

fn color(k: Kind) -> &'static str {
    match permille(k) {
        Some(990..) => "brightgreen",
        Some(900..) => "green",
        Some(750..) => "yellowgreen",
        Some(500..) => "yellow",
        Some(_) | None => "orange",
    }
}

/// `12345` as `12,345`.
fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn render(m: &Measurement) -> String {
    let Measurement {
        schema: _,
        complete,
        host_lanes,
        vertex_pixel,
        compute,
        compute_wave64,
        mesh,
        commit,
        commit_date,
    } = m;
    // A host with fewer lanes refuses what was emitted for 64-lane waves.
    let refused = if *host_lanes < 64 { *compute_wave64 } else { 0 };
    let compute = Kind { found: compute.found, emitted: compute.emitted.saturating_sub(refused) };
    let kinds = [("Mesh", *mesh), ("Vertex/pixel", *vertex_pixel), ("Compute", compute)];
    let all = Kind {
        found: kinds.iter().map(|(_, k)| k.found).sum(),
        emitted: kinds.iter().map(|(_, k)| k.emitted).sum(),
    };
    let row = |name: &str, k: Kind| {
        format!(
            "| {name} | {} of {} | {} | {} |\n",
            thousands(k.emitted),
            thousands(k.found),
            percent(k),
            bar(k)
        )
    };
    let mut s = format!(
        "<!-- Generated by `cargo xtask progress` from {MEASUREMENT}; edit that file, not this block. -->\n\
         ![shaders translated: {p}](https://img.shields.io/badge/shaders%20translated-{e}-{c})\n\n\
         | Programs | Translated for a {host_lanes}-lane GPU | Share | |\n|---|---:|---:|---|\n",
        p = percent(all),
        e = percent(all).replace('%', "%25"),
        c = color(all),
    );
    s += &row("**All**", all);
    for (name, k) in kinds {
        s += &row(name, k);
    }
    s += &format!(
        "\nMeasured with `victory translate --host-lanes {host_lanes} --json` over the game \
         directory (files above 2 GiB are not read), built at commit `{commit}` \
         ({commit_date}). A program counts once it is emitted as DXIL that a GPU with \
         {host_lanes}-lane waves accepts; validation and pipeline creation are not part of the \
         share.\n"
    );
    if refused > 0 {
        s += &format!(
            "{} more compute programs are emitted for 64-lane waves, which such a GPU refuses.\n",
            thousands(refused)
        );
    }
    if !complete {
        s += "\n**Incomplete run:** some files were not measured, so every count is a lower bound.\n";
    }
    s
}

/// `readme` with the text between [`START`] and [`END`] replaced by `block`.
fn splice(readme: &str, block: &str) -> Result<String> {
    let (head, rest) =
        readme.split_once(START).with_context(|| format!("README.md has no `{START}` line"))?;
    let (_, tail) = rest
        .split_once(END)
        .with_context(|| format!("README.md has no `{END}` after `{START}`"))?;
    Ok(format!("{head}{START}\n{block}{END}{tail}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(found: usize, emitted: usize) -> Kind {
        Kind { found, emitted }
    }

    fn sample(complete: bool) -> Measurement {
        Measurement {
            schema: SCHEMA,
            complete,
            host_lanes: 32,
            vertex_pixel: k(12_346, 10_617),
            compute: k(2_539, 1_643),
            compute_wave64: 0,
            mesh: k(8_506, 8_506),
            commit: "0123456789".into(),
            commit_date: "2026-11-25".into(),
        }
    }

    #[test]
    fn readme_progress_matches_the_measurement() {
        run(&crate::workspace_root().unwrap(), None, true).unwrap();
    }

    #[test]
    fn a_share_short_of_everything_never_shows_100() {
        assert_eq!(percent(k(10_000, 9_999)), "99.9%");
        assert_eq!(percent(k(8, 8)), "100.0%");
        assert_eq!(percent(k(0, 0)), "–");
        assert_eq!(bar(k(10, 5)), format!("`{}{}`", "█".repeat(10), "░".repeat(10)));
        assert_eq!(bar(k(0, 0)), format!("`{}`", "░".repeat(20)));
    }

    #[test]
    fn thousands_are_grouped() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1_000), "1,000");
        assert_eq!(thousands(1_234_567), "1,234,567");
    }

    #[test]
    fn the_block_sums_every_kind_and_names_the_commit() {
        let block = render(&sample(true));
        assert!(block.contains("| **All** | 20,766 of 23,391 | 88.7% |"), "{block}");
        assert!(block.contains("| Mesh | 8,506 of 8,506 | 100.0% |"), "{block}");
        assert!(block.contains("shaders%20translated-88.7%25-yellowgreen"), "{block}");
        assert!(block.contains("`0123456789` (2026-11-25)"), "{block}");
        assert!(block.contains("--host-lanes 32 --json"), "{block}");
        assert!(!block.contains("Incomplete"), "{block}");
        assert!(render(&sample(false)).contains("**Incomplete run:**"));
    }

    #[test]
    fn wave64_compute_counts_only_on_a_64_lane_host() {
        let m = |host_lanes| Measurement { host_lanes, compute_wave64: 100, ..sample(true) };
        let narrow = render(&m(32));
        assert!(narrow.contains("| Compute | 1,543 of 2,539 |"), "{narrow}");
        assert!(narrow.contains("100 more compute programs"), "{narrow}");
        assert!(narrow.contains("Translated for a 32-lane GPU"), "{narrow}");
        let wide = render(&m(64));
        assert!(wide.contains("| Compute | 1,643 of 2,539 |"), "{wide}");
        assert!(!wide.contains("more compute programs"), "{wide}");
    }

    #[test]
    fn splice_replaces_only_the_block() {
        let readme = format!("# T\n\n{START}\nold\n{END}\n\n## Next\n");
        assert_eq!(
            splice(&readme, "new\n").unwrap(),
            format!("# T\n\n{START}\nnew\n{END}\n\n## Next\n")
        );
        assert!(splice("# T\n", "new\n").is_err());
        assert!(splice(&format!("{START}\nno end\n"), "new\n").is_err());
    }

    #[test]
    fn a_summary_of_another_schema_is_refused() {
        let summary = |schema| {
            format!(
                r#"{{"schema":{schema},"complete":true,"host_lanes":32,"vertex_pixel":{{"found":2,"emitted":1}},
                "compute":{{"found":0,"emitted":0}},"compute_wave64":0,"mesh":{{"found":1,"emitted":1}}}}"#
            )
        };
        let m = measured(&summary(SCHEMA), "c".into(), "d".into()).unwrap();
        assert_eq!((m.vertex_pixel, m.commit.as_str()), (k(2, 1), "c"));
        assert!(measured(&summary(SCHEMA + 1), "c".into(), "d".into()).is_err());
    }
}

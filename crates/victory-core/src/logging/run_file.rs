//! The run's log file: its name, its creation, the removal of old ones,
//! and the crash line written beside the subscriber.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::time::{FormatTime, SystemTime};

use super::SUMMARY;

/// Run files kept in their directory, the new one included.
pub const KEEP_RUNS: usize = 20;

/// This process's run file, once [`super::init_run`] made it.
static RUN_FILE: OnceLock<PathBuf> = OnceLock::new();

pub(super) fn remember(path: PathBuf) {
    // A second `init_run` in one process keeps the first file.
    if RUN_FILE.set(path).is_err() {
        tracing::warn!("a second run log was opened; crashes go to the first");
    }
}

/// The time as the JSON lines write it (RFC 3339, UTC).
fn now_rfc3339() -> String {
    let mut text = String::new();
    if SystemTime.format_time(&mut Writer::new(&mut text)).is_err() {
        text.clear();
    }
    text
}

/// The current UTC time as a file name part.
pub(super) fn now_stamp() -> String {
    file_stamp(&now_rfc3339())
}

/// `2026-11-25T12:00:01.123456Z` → `20261125T120001Z`: whole seconds, no
/// characters a Windows file name refuses.
pub(super) fn file_stamp(rfc3339: &str) -> String {
    let seconds = rfc3339.split('.').next().unwrap_or(rfc3339);
    let mut stamp: String = seconds.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    if !stamp.ends_with('Z') {
        stamp.push('Z');
    }
    stamp
}

/// Run file names sort by time.
pub(super) fn run_file_name(stamp: &str, pid: u32) -> String {
    format!("run-{stamp}-{pid}.jsonl")
}

fn is_run_file(name: &str) -> bool {
    name.starts_with("run-") && name.ends_with(".jsonl")
}

/// Creates the run file in `dir`, opened to append: every write lands at
/// the end, even with [`append_crash`] writing through another handle.
pub(super) fn open(dir: &Path, stamp: &str, pid: u32) -> std::io::Result<(PathBuf, File)> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(run_file_name(stamp, pid));
    let file = OpenOptions::new().create(true).append(true).open(&path)?;
    Ok((path, file))
}

/// What [`prune`] removed and failed to remove.
pub(super) struct Pruned {
    pub(super) removed: Vec<PathBuf>,
    pub(super) failed: Vec<(PathBuf, std::io::Error)>,
}

/// Removes all but the newest `keep` run files of `dir`; other files stay.
pub(super) fn prune(dir: &Path, keep: usize) -> Pruned {
    let mut pruned = Pruned { removed: Vec::new(), failed: Vec::new() };
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            pruned.failed.push((dir.to_owned(), e));
            return pruned;
        }
    };
    let mut names = Vec::new();
    for entry in entries {
        match entry {
            Ok(entry) => {
                let name = entry.file_name().to_string_lossy().into_owned();
                if is_run_file(&name) {
                    names.push(name);
                }
            }
            Err(e) => pruned.failed.push((dir.to_owned(), e)),
        }
    }
    names.sort_unstable_by(|a, b| b.cmp(a));
    for name in names.into_iter().skip(keep) {
        let path = dir.join(name);
        match std::fs::remove_file(&path) {
            Ok(()) => pruned.removed.push(path),
            Err(e) => pruned.failed.push((path, e)),
        }
    }
    pruned
}

/// A crash as a line of the run file, with the keys the subscriber writes.
pub(super) fn crash_line(report: &str) -> String {
    let thread = std::thread::current();
    let mut line = serde_json::Map::new();
    line.insert("timestamp".into(), now_rfc3339().into());
    line.insert("level".into(), "ERROR".into());
    line.insert("target".into(), SUMMARY.into());
    line.insert("threadId".into(), format!("{:?}", thread.id()).into());
    if let Some(name) = thread.name() {
        line.insert("threadName".into(), name.into());
    }
    line.insert("message".into(), "crash".into());
    line.insert("report".into(), report.into());
    serde_json::Value::Object(line).to_string()
}

/// Appends `report` to this process's run file, if it has one. It opens
/// the file anew rather than going through the subscriber, which the
/// crashing thread may be in the middle of.
pub fn append_crash(report: &str) -> std::io::Result<()> {
    match RUN_FILE.get() {
        Some(path) => append_crash_to(path, report),
        None => Ok(()),
    }
}

pub(super) fn append_crash_to(path: &Path, report: &str) -> std::io::Result<()> {
    let mut line = crash_line(report);
    line.push('\n');
    OpenOptions::new().append(true).open(path)?.write_all(line.as_bytes())
}

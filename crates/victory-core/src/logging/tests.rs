use std::collections::BTreeSet;
use std::ffi::OsString;
use std::io::Write;
use std::sync::{Arc, Mutex};

use super::run_file::{append_crash_to, crash_line, file_stamp, open, prune, run_file_name};
use super::*;

/// An in-memory destination standing in for the console or the run file.
#[derive(Clone, Default)]
struct Buf(Arc<Mutex<Vec<u8>>>);

impl Write for Buf {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Buf {
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
    }
    fn lines(&self) -> Vec<serde_json::Value> {
        self.text().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
    }
}

/// Runs `body` under a subscriber built from `spec` and returns what the
/// console and the run file received.
fn logged(spec: &Spec, body: impl FnOnce()) -> (Buf, Buf) {
    let (console, file) = (Buf::default(), Buf::default());
    let (c, f) = (console.clone(), file.clone());
    let subscriber = subscriber(spec, move || c.clone(), Some(move || f.clone()));
    tracing::subscriber::with_default(subscriber, body);
    (console, file)
}

fn keys(value: &serde_json::Value) -> BTreeSet<String> {
    value.as_object().unwrap().keys().cloned().collect()
}

#[test]
fn a_file_line_is_flat_json_with_utc_time_and_thread() {
    let spec = Spec::new(None, "info");
    let (_, file) = logged(&spec, || tracing::info!(hash = "ab12", stage = "Ps", "shader"));
    let lines = file.lines();
    assert_eq!(lines.len(), 1, "{}", file.text());
    let line = &lines[0];
    assert_eq!(line["message"], "shader");
    assert_eq!(line["hash"], "ab12");
    assert_eq!(line["level"], "INFO");
    assert!(line.get("fields").is_none(), "fields are flattened: {line}");
    assert!(line["threadId"].as_str().unwrap().starts_with("ThreadId("));
    let time = line["timestamp"].as_str().unwrap();
    assert!(time.ends_with('Z') && time.contains('T'), "{time}");
}

#[test]
fn summary_events_pass_a_warn_filter_in_both_sinks() {
    let spec = Spec::new(Some(Ok("warn".into())), "info");
    let (console, file) = logged(&spec, || {
        tracing::info!(target: SUMMARY, "run_start");
        tracing::info!("ordinary");
        tracing::warn!("trouble");
    });
    for text in [console.text(), file.text()] {
        assert!(text.contains("run_start"), "{text}");
        assert!(text.contains("trouble"), "{text}");
        assert!(!text.contains("ordinary"), "{text}");
    }
}

#[test]
fn other_events_follow_the_filter() {
    let spec = Spec::new(Some(Ok("info,noisy=warn".into())), "info");
    let (console, file) = logged(&spec, || {
        tracing::info!(target: "noisy", "hidden");
        tracing::warn!(target: "noisy", "shown");
        tracing::info!(target: "quiet", "kept");
    });
    for text in [console.text(), file.text()] {
        assert!(!text.contains("hidden") && text.contains("shown") && text.contains("kept"));
    }
}

#[test]
fn an_invalid_filter_falls_back_and_is_reported() {
    let spec = Spec::new(Some(Ok("victory=loud".into())), "info");
    assert!(spec.refused().is_some_and(|why| why.contains("victory=loud")));
    let (console, _) = logged(&spec, || tracing::info!("still logged"));
    assert!(console.text().contains("still logged"));
    let unreadable = Spec::new(Some(Err("not unicode".into())), "info");
    assert!(unreadable.refused().is_some());
    assert!(Spec::new(None, "info").refused().is_none());
}

#[test]
fn run_file_names_sort_by_time_and_carry_the_pid() {
    assert_eq!(file_stamp("2026-11-25T12:00:01.123456Z"), "20261125T120001Z");
    let early = run_file_name("20261125T120001Z", 99_999);
    let late = run_file_name("20261125T120002Z", 7);
    assert_eq!(early, "run-20261125T120001Z-99999.jsonl");
    assert!(early < late);
}

#[test]
fn pruning_keeps_the_newest_runs_and_ignores_other_files() {
    let dir = tempfile::tempdir().unwrap();
    for second in 10..35 {
        let name = run_file_name(&format!("20261125T1200{second}Z"), 1);
        std::fs::write(dir.path().join(name), b"{}\n").unwrap();
    }
    std::fs::write(dir.path().join("notes.txt"), b"mine").unwrap();
    let pruned = prune(dir.path(), 20);
    assert!(pruned.failed.is_empty(), "{:?}", pruned.failed);
    assert_eq!(pruned.removed.len(), 5);
    let mut left: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    left.sort();
    assert_eq!(left.len(), 21);
    let oldest_kept = left.iter().find(|name| name.starts_with("run-")).unwrap();
    assert_eq!(*oldest_kept, run_file_name("20261125T120015Z", 1));
    assert!(left.contains(&"notes.txt".to_owned()));
}

#[test]
fn a_crash_line_has_the_keys_of_an_event_line() {
    let spec = Spec::new(None, "info");
    let (_, file) = logged(&spec, || tracing::error!(target: SUMMARY, report = "boom", "crash"));
    let event = &file.lines()[0];
    let crash: serde_json::Value = serde_json::from_str(&crash_line("boom")).unwrap();
    assert_eq!(keys(event), keys(&crash));
    for key in ["level", "target", "message", "report", "threadId"] {
        assert_eq!(event[key], crash[key], "{key}");
    }
}

#[test]
fn a_crash_is_appended_after_the_lines_already_written() {
    let dir = tempfile::tempdir().unwrap();
    let (path, mut file) = open(dir.path(), "20261125T120000Z", 1).unwrap();
    file.write_all(b"{\"message\":\"before\"}\n").unwrap();
    append_crash_to(&path, "boom").unwrap();
    file.write_all(b"{\"message\":\"after\"}\n").unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let messages: Vec<String> = text
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["message"].to_string())
        .collect();
    assert_eq!(messages, ["\"before\"", "\"crash\"", "\"after\""]);
}

#[test]
fn append_crash_without_a_run_file_does_nothing() {
    // No `init_run` in this test binary, so there is no run file.
    assert!(append_crash("boom").is_ok());
}

#[test]
fn switches_keep_victory_and_rust_log_only() {
    let vars = [
        ("VICTORY_SKIP_PIPELINES", "ab12"),
        ("PATH", "C:\\Windows"),
        ("RUST_LOG", "warn"),
        ("VICTORY_LOG", "info"),
        ("VICTORYX", "no"),
    ]
    .map(|(k, v)| (OsString::from(k), OsString::from(v)));
    let kept = switches(vars.into_iter());
    let names: Vec<&str> = kept.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(names, ["RUST_LOG", "VICTORY_LOG", "VICTORY_SKIP_PIPELINES"]);
    assert_eq!(kept[2].1, "ab12");
}

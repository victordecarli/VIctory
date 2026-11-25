//! What a process of `victory run` records about itself at the start of
//! its log, so a reader knows which build ran, as what, and with which
//! switches.

use victory_core::logging::{SUMMARY, switches};

/// Logs `run_start`: this process's `role` (`run`, `audio-capture`), its id,
/// the one it replaced (`previous_pid`, after a relaunch) and the one that
/// started it to run beside it (`parent_pid`, the audio recorder's), the
/// BLAKE3 of our executable (which build this is), whether debugging aids
/// are built in, the arguments, and every `VICTORY_*` and `RUST_LOG`
/// variable (the last two as JSON text). A field that is `None` is left out.
pub fn start(role: &str, previous_pid: Option<u32>, parent_pid: Option<u32>) {
    let exe = match std::env::current_exe().and_then(std::fs::read) {
        Ok(bytes) => blake3::hash(&bytes).to_hex().to_string(),
        Err(e) => {
            tracing::warn!(error = %e, "cannot read our executable to name this build");
            "unknown".to_owned()
        }
    };
    // As JSON text, which a reader can parse (`jq '.switches | fromjson'`).
    let args: Vec<String> =
        std::env::args_os().map(|arg| arg.to_string_lossy().into_owned()).collect();
    let args = serde_json::Value::from(args);
    let switches: serde_json::Map<String, serde_json::Value> =
        switches(std::env::vars_os()).into_iter().map(|(k, v)| (k, v.into())).collect();
    let switches = serde_json::Value::Object(switches);
    tracing::info!(
        target: SUMMARY,
        role,
        pid = std::process::id(),
        previous_pid,
        parent_pid,
        exe_blake3 = %exe,
        diagnostics = cfg!(feature = "diagnostics"),
        %args,
        %switches,
        "run_start"
    );
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    /// A run log file in memory.
    #[derive(Clone, Default)]
    struct Memory(Arc<Mutex<Vec<u8>>>);

    impl Write for Memory {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap_or_else(|e| e.into_inner()).extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// The `run_start` line [`super::start`] writes to the run file.
    fn run_start(
        role: &str,
        previous_pid: Option<u32>,
        parent_pid: Option<u32>,
    ) -> serde_json::Value {
        let file = Memory::default();
        let writer = file.clone();
        let spec = victory_core::logging::Spec::new(None, "info");
        let subscriber =
            victory_core::logging::subscriber(&spec, std::io::sink, Some(move || writer.clone()));
        tracing::subscriber::with_default(subscriber, || {
            super::start(role, previous_pid, parent_pid);
        });
        let text = String::from_utf8(file.0.lock().unwrap().clone()).unwrap();
        let line = text.lines().find(|l| l.contains(r#""message":"run_start""#)).unwrap();
        serde_json::from_str(line).unwrap()
    }

    #[test]
    fn the_audio_recorder_names_the_process_that_started_it() {
        let line = run_start("audio-capture", None, Some(42));
        assert_eq!(line["role"], "audio-capture");
        assert_eq!(line["parent_pid"], 42);
        assert!(line.get("previous_pid").is_none(), "{line}");
    }

    #[test]
    fn a_relaunched_run_names_the_process_it_replaced_not_a_parent() {
        let line = run_start("run", Some(7), None);
        assert_eq!(line["previous_pid"], 7);
        assert!(line.get("parent_pid").is_none(), "{line}");
    }
}

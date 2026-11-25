//! What a process of `victory run` records about itself at the start of
//! its log, so a reader knows which build ran, as what, and with which
//! switches.

use victory_core::logging::{SUMMARY, switches};

/// Logs `run_start`: this process's `role` (`run`, `audio-capture`), its id
/// and the one it replaced (`previous_pid`, after a relaunch), the BLAKE3 of
/// our executable (which build this is), whether debugging aids are built
/// in, the arguments, and every `VICTORY_*` and `RUST_LOG` variable (the
/// last two as JSON text).
pub fn start(role: &str, previous_pid: Option<u32>) {
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
        exe_blake3 = %exe,
        diagnostics = cfg!(feature = "diagnostics"),
        %args,
        %switches,
        "run_start"
    );
}

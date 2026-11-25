//! Warnings logged once, and counted.
//!
//! A case the crate skips or does not implement yet logs one `warn` line
//! the first time it happens ([`warn_once!`]). Every later case adds to the
//! warning's total instead, and [`report`], with every frame report, logs
//! the total of each warning that had new cases since the last report
//! (`"warning total"`, target `victory::summary`), so a run's log says how
//! often a case happened, not only that it did.
//!
//! Counting is atomic and allocates nothing; only a warning's first case
//! touches the registry.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// Every warning that had a case, in the order of their first.
static REGISTRY: Mutex<Vec<&'static Warning>> = Mutex::new(Vec::new());

/// One warning: where it comes from, what it says, and how often it
/// happened. Made `static`, one per place in the code or per fixed key.
#[cfg_attr(
    not(any(windows, test)),
    allow(
        dead_code,
        reason = "only `changed` and the Windows report read the site, the message and what was reported"
    )
)]
pub(crate) struct Warning {
    site: &'static str,
    message: &'static str,
    total: AtomicU64,
    /// The total as of the last report.
    reported: AtomicU64,
    registered: AtomicBool,
    /// What tells this warning's key apart, kept from its first case.
    detail: OnceLock<String>,
}

impl Warning {
    /// A warning with no case yet, at `site` ("file:line").
    pub(crate) const fn new(site: &'static str, message: &'static str) -> Self {
        Self {
            site,
            message,
            total: AtomicU64::new(0),
            reported: AtomicU64::new(0),
            registered: AtomicBool::new(false),
            detail: OnceLock::new(),
        }
    }

    /// Counts a case; `first` runs for the warning's first only (the log
    /// line).
    pub(crate) fn hit(&'static self, first: impl FnOnce()) {
        self.total.fetch_add(1, Ordering::Relaxed);
        if !self.registered.swap(true, Ordering::Relaxed) {
            self.register();
            first();
        }
    }

    /// As [`Warning::hit`], keeping `detail()` from the first case for
    /// the report: the object or slot a warning with a fixed key is about.
    #[allow(dead_code, reason = "the warnings with a fixed key use it from the next change on")]
    pub(crate) fn hit_with(&'static self, detail: impl FnOnce() -> String, first: impl FnOnce()) {
        self.total.fetch_add(1, Ordering::Relaxed);
        if !self.registered.swap(true, Ordering::Relaxed) {
            // Set before it is listed, so a report never sees it missing.
            if self.detail.set(detail()).is_err() {
                tracing::debug!(site = self.site, "a warning's detail was already set");
            }
            self.register();
            first();
        }
    }

    fn register(&'static self) {
        REGISTRY.lock().unwrap_or_else(|e| e.into_inner()).push(self);
    }
}

/// Logs `message` as a `warn` the first time this place runs, and counts
/// every time. The message is a literal, first; the rest are the
/// `tracing` fields of the line:
/// `warn_once!("a draw dropped", slot, kind = ?kind)`.
macro_rules! warn_once {
    ($message:literal $(, $($field:tt)+)?) => {{
        static WARNING: $crate::warn::Warning =
            $crate::warn::Warning::new(concat!(file!(), ":", line!()), $message);
        WARNING.hit(|| tracing::warn!($($($field)+,)? $message));
    }};
}
pub(crate) use warn_once;

/// A warning with new cases since the last report.
#[cfg(any(windows, test))]
#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
struct Changed {
    warning: &'static Warning,
    total: u64,
    since_last: u64,
}

#[cfg(test)]
impl std::fmt::Debug for Warning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.site)
    }
}

#[cfg(test)]
impl PartialEq for Warning {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

#[cfg(test)]
impl Eq for Warning {}

/// The `items` with cases since the last call, in their order; marks what
/// it returns as reported.
#[cfg(any(windows, test))]
fn changed(items: &[&'static Warning]) -> Vec<Changed> {
    let mut out = Vec::new();
    for &warning in items {
        let total = warning.total.load(Ordering::Relaxed);
        let before = warning.reported.swap(total, Ordering::Relaxed);
        if total > before {
            out.push(Changed { warning, total, since_last: total - before });
        }
    }
    out
}

/// Logs the total of every warning that had cases since the last call.
/// Nothing if none did.
#[cfg(windows)]
pub(crate) fn report() {
    let all = REGISTRY.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let frame = crate::frame::current();
    for Changed { warning, total, since_last } in changed(&all) {
        let detail = warning.detail.get().map_or("", String::as_str);
        tracing::info!(
            target: crate::SUMMARY,
            site = warning.site,
            message = warning.message,
            detail,
            total,
            since_last,
            frame,
            "warning total"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn listed(w: &'static Warning) -> usize {
        let all = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
        all.iter().filter(|&&x| std::ptr::eq(x, w)).count()
    }

    #[test]
    fn a_warning_is_listed_and_logged_once() {
        static W: Warning = Warning::new("warn.rs:1", "listed once");
        let mut logged = 0;
        for _ in 0..3 {
            W.hit(|| logged += 1);
        }
        assert_eq!(logged, 1);
        assert_eq!(listed(&W), 1);
    }

    #[test]
    fn every_case_adds_to_the_total() {
        static W: Warning = Warning::new("warn.rs:2", "totals");
        for _ in 0..5 {
            W.hit(|| {});
        }
        assert_eq!(W.total.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn changed_gives_the_new_cases_and_skips_a_warning_that_stopped() {
        static A: Warning = Warning::new("warn.rs:3", "a");
        static B: Warning = Warning::new("warn.rs:4", "b");
        static C: Warning = Warning::new("warn.rs:5", "c");
        A.hit(|| {});
        A.hit(|| {});
        B.hit(|| {});
        let items = [&A, &B, &C];
        assert_eq!(
            changed(&items),
            [
                Changed { warning: &A, total: 2, since_last: 2 },
                Changed { warning: &B, total: 1, since_last: 1 },
            ]
        );
        assert_eq!(changed(&items), []);
        A.hit(|| {});
        assert_eq!(changed(&items), [Changed { warning: &A, total: 3, since_last: 1 }]);
    }

    #[test]
    fn the_detail_stays_with_the_first_case() {
        static W: Warning = Warning::new("warn.rs:6", "detail");
        let mut logged = Vec::new();
        W.hit_with(|| "first".into(), || logged.push("first"));
        W.hit_with(|| "second".into(), || logged.push("second"));
        assert_eq!(logged, ["first"]);
        assert_eq!(W.detail.get().map(String::as_str), Some("first"));
        assert_eq!(W.total.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn the_macro_makes_one_warning_per_place_and_counts_its_cases() {
        for slot in 0..4 {
            warn_once!("a case of the macro", slot);
        }
        let all = REGISTRY.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let mine: Vec<_> = all.iter().filter(|w| w.message == "a case of the macro").collect();
        assert_eq!(mine.len(), 1);
        assert_eq!(mine[0].total.load(Ordering::Relaxed), 4);
        assert!(mine[0].site.contains("warn.rs:"));
    }
}

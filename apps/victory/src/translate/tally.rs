//! What a translation run counted, and what kept it from counting
//! everything.

use std::any::Any;
use std::collections::BTreeMap;
use std::path::Path;

use crate::walk::Skip;

/// Reason → occurrences and the first example.
pub(super) type Reasons = BTreeMap<String, (usize, String)>;

#[derive(Default)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub(super) struct Tally {
    /// Files read whole.
    pub(super) files: usize,
    /// Files above the size limit: not read, and not a failure, since a
    /// game directory can hold archives that large.
    pub(super) oversized: usize,
    pub(super) containers: usize,
    pub(super) duplicates: usize,
    pub(super) programs: usize,
    pub(super) translated: usize,
    pub(super) instructions: usize,
    pub(super) structured: usize,
    pub(super) vector: usize,
    pub(super) guarded: usize,
    /// Guards that read a mask the pass cannot see into.
    pub(super) opaque: usize,
    pub(super) lane_branches: usize,
    pub(super) uniform_branches: usize,
    pub(super) tracked: usize,
    pub(super) loaded: usize,
    pub(super) built: usize,
    pub(super) constant: usize,
    /// Descriptors and scalar accesses with a computed address term.
    pub(super) dynamic: usize,
    pub(super) scalar: usize,
    pub(super) used: usize,
    /// Used-resource entries whose root parameter no descriptor traces
    /// to, by resource and root type.
    pub(super) untraced_used: BTreeMap<String, usize>,
    /// Root parameters descriptors trace to that no used entry names.
    pub(super) unlisted: usize,
    /// Vertex and pixel programs, and those whose I/O was raised.
    pub(super) io_programs: usize,
    pub(super) io_raised: usize,
    pub(super) interpolated: usize,
    pub(super) vertex_values: usize,
    /// Exports by target kind.
    pub(super) exports: BTreeMap<&'static str, usize>,
    /// Exports still packed as f16 pairs after the export patch.
    pub(super) compressed: usize,
    /// Vertex and pixel programs every earlier stage handled, and how far
    /// emission got with them.
    pub(super) emit_programs: usize,
    pub(super) emitted: usize,
    pub(super) validated: usize,
    pub(super) pipelines: usize,
    /// The same for compute programs.
    pub(super) cs_programs: usize,
    pub(super) cs_emitted: usize,
    pub(super) cs_validated: usize,
    pub(super) cs_pipelines: usize,
    /// Compute programs emitted that require 64-lane waves (`WaveSize(64)`):
    /// a host whose waves have 32 lanes (NVIDIA) refuses their pipelines.
    pub(super) cs_wave64: usize,
    /// The same for mesh programs.
    pub(super) ms_programs: usize,
    pub(super) ms_emitted: usize,
    pub(super) ms_validated: usize,
    pub(super) ms_pipelines: usize,
    /// What the translator cannot handle yet: the measurement itself.
    pub(super) failed: Reasons,
    /// What kept the run from measuring everything (files not read,
    /// panics): any entry makes every count above a lower bound.
    pub(super) infra: Reasons,
    /// With `--list`: one line per program and outcome.
    pub(super) list: bool,
    pub(super) lines: Vec<String>,
}

impl Tally {
    pub(super) fn new(list: bool) -> Self {
        Self { list, ..Self::default() }
    }

    pub(super) fn fail(&mut self, reason: String, example: impl FnOnce() -> String) {
        let lines = self.list.then_some(&mut self.lines);
        note(&mut self.failed, lines, reason, example);
    }

    pub(super) fn infra(&mut self, reason: String, example: impl FnOnce() -> String) {
        let lines = self.list.then_some(&mut self.lines);
        note(&mut self.infra, lines, reason, example);
    }

    /// Records a file the walk could not hand over: an oversized one is
    /// counted and logged, anything else is an infrastructure failure.
    pub(super) fn unread(&mut self, path: &Path, skip: Skip) {
        match skip {
            Skip::TooLarge => {
                tracing::warn!(path = %path.display(), "file above the size limit skipped");
                self.oversized += 1;
            }
            Skip::Unreadable(_) | Skip::Walk(_) => {
                self.infra(skip.reason(), || path.display().to_string());
            }
        }
    }

    /// Runs `body` on a tally of its own, merged only if it returns: a
    /// panic halfway would otherwise leave a program counted as translated
    /// but never structured, and the stages would stop adding up.
    pub(super) fn isolate(
        &mut self,
        example: impl FnOnce() -> String,
        body: impl FnOnce(&mut Tally),
    ) {
        let mut own = Tally::new(self.list);
        // Unwind safety: on a panic `own` is dropped unread, and what the
        // body shares (the dedup set, kept shaders) stays valid as is.
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(&mut own))) {
            Ok(()) => self.merge(own),
            Err(payload) => self.infra(panic_reason(&*payload), example),
        }
    }

    /// `Err` if the run did not measure everything it was given.
    pub(super) fn verdict(&self) -> anyhow::Result<()> {
        let n: usize = self.infra.values().map(|r| r.0).sum();
        anyhow::ensure!(
            n == 0,
            "{n} infrastructure failure(s): every count above is a lower bound"
        );
        Ok(())
    }

    pub(super) fn merge(&mut self, other: Tally) {
        // Destructured in full so that a new field cannot be left out.
        let Tally {
            files,
            oversized,
            containers,
            duplicates,
            programs,
            translated,
            instructions,
            structured,
            vector,
            guarded,
            opaque,
            lane_branches,
            uniform_branches,
            tracked,
            loaded,
            built,
            constant,
            dynamic,
            scalar,
            used,
            untraced_used,
            unlisted,
            io_programs,
            io_raised,
            interpolated,
            vertex_values,
            exports,
            compressed,
            emit_programs,
            emitted,
            validated,
            pipelines,
            cs_programs,
            cs_emitted,
            cs_validated,
            cs_pipelines,
            cs_wave64,
            ms_programs,
            ms_emitted,
            ms_validated,
            ms_pipelines,
            failed,
            infra,
            list: _,
            lines,
        } = other;
        self.files += files;
        self.oversized += oversized;
        self.containers += containers;
        self.duplicates += duplicates;
        self.programs += programs;
        self.translated += translated;
        self.instructions += instructions;
        self.structured += structured;
        self.vector += vector;
        self.guarded += guarded;
        self.opaque += opaque;
        self.lane_branches += lane_branches;
        self.uniform_branches += uniform_branches;
        self.tracked += tracked;
        self.loaded += loaded;
        self.built += built;
        self.constant += constant;
        self.dynamic += dynamic;
        self.scalar += scalar;
        self.used += used;
        for (k, n) in untraced_used {
            *self.untraced_used.entry(k).or_default() += n;
        }
        self.unlisted += unlisted;
        self.io_programs += io_programs;
        self.io_raised += io_raised;
        self.interpolated += interpolated;
        self.vertex_values += vertex_values;
        for (k, n) in exports {
            *self.exports.entry(k).or_default() += n;
        }
        self.compressed += compressed;
        self.emit_programs += emit_programs;
        self.emitted += emitted;
        self.validated += validated;
        self.pipelines += pipelines;
        self.cs_programs += cs_programs;
        self.cs_emitted += cs_emitted;
        self.cs_validated += cs_validated;
        self.cs_pipelines += cs_pipelines;
        self.cs_wave64 += cs_wave64;
        self.ms_programs += ms_programs;
        self.ms_emitted += ms_emitted;
        self.ms_validated += ms_validated;
        self.ms_pipelines += ms_pipelines;
        add(&mut self.failed, failed);
        add(&mut self.infra, infra);
        self.lines.extend(lines);
    }
}

fn add(into: &mut Reasons, from: Reasons) {
    for (reason, (n, example)) in from {
        into.entry(reason).or_insert((0, example)).0 += n;
    }
}

fn note(
    reasons: &mut Reasons,
    lines: Option<&mut Vec<String>>,
    reason: String,
    example: impl FnOnce() -> String,
) {
    match lines {
        Some(lines) => {
            let e = example();
            lines.push(format!("{e}  {reason}"));
            reasons.entry(reason).or_insert((0, e)).0 += 1;
        }
        None => reasons.entry(reason).or_insert_with(|| (0, example())).0 += 1,
    }
}

fn panic_reason(payload: &(dyn Any + Send)) -> String {
    let text = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("(no message)");
    // Values in the message (lengths, indices, offsets) would make one
    // reason per shader: each run of digits is masked so equal panics group.
    let mut reason = String::from("panic: ");
    for c in text.lines().next().unwrap_or("").chars() {
        if !c.is_ascii_digit() {
            reason.push(c);
        } else if !reason.ends_with('#') {
            reason.push('#');
        }
    }
    reason
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reasons(entries: &[(&str, usize, &str)]) -> Reasons {
        entries.iter().map(|&(r, n, e)| (r.to_owned(), (n, e.to_owned()))).collect()
    }

    #[test]
    fn translator_and_infrastructure_failures_are_kept_apart() {
        let mut t = Tally::default();
        t.fail("unsupported: v_foo".into(), || "a".into());
        t.infra("file: too large".into(), || "b".into());
        assert_eq!(t.failed, reasons(&[("unsupported: v_foo", 1, "a")]));
        assert_eq!(t.infra, reasons(&[("file: too large", 1, "b")]));
    }

    #[test]
    fn an_oversized_file_is_counted_not_a_failure() {
        let mut t = Tally::default();
        t.unread(Path::new("big.pak"), Skip::TooLarge);
        assert_eq!(t.oversized, 1);
        assert!(t.infra.is_empty());
        assert!(t.verdict().is_ok());
    }

    #[test]
    fn an_unreadable_file_is_an_infrastructure_failure() {
        let mut t = Tally::default();
        t.unread(Path::new("a"), Skip::Unreadable(std::io::ErrorKind::NotFound));
        assert_eq!(t.infra, reasons(&[("file: unreadable: NotFound", 1, "a")]));
        assert!(t.verdict().is_err());
    }

    #[test]
    fn translator_failures_alone_are_a_complete_run() {
        let mut t = Tally::default();
        for _ in 0..1000 {
            t.fail("unsupported: v_foo".into(), String::new);
        }
        assert!(t.verdict().is_ok());
    }

    #[test]
    fn one_infrastructure_failure_makes_the_run_incomplete() {
        let mut t = Tally::default();
        t.fail("unsupported: v_foo".into(), String::new);
        t.infra("file: unreadable: PermissionDenied".into(), String::new);
        let err = t.verdict().unwrap_err().to_string();
        assert!(err.contains("1 infrastructure failure"), "{err}");
    }

    #[test]
    fn list_mode_lists_infrastructure_failures_too() {
        let mut t = Tally::new(true);
        t.infra("walk: NotFound".into(), || "/games/x".into());
        assert_eq!(t.lines, ["/games/x  walk: NotFound"]);
    }

    #[test]
    fn merge_sums_counts_and_keeps_the_first_example() {
        let mut a = Tally { files: 1, ms_pipelines: 2, ..Tally::default() };
        a.fail("r".into(), || "a".into());
        a.infra("i".into(), || "a".into());
        let mut b = Tally { files: 3, ms_pipelines: 4, ..Tally::default() };
        b.fail("r".into(), || "b".into());
        b.infra("i".into(), || "b".into());
        *b.exports.entry("color").or_default() += 5;
        a.merge(b);
        assert_eq!((a.files, a.ms_pipelines), (4, 6));
        assert_eq!(a.failed, reasons(&[("r", 2, "a")]));
        assert_eq!(a.infra, reasons(&[("i", 2, "a")]));
        assert_eq!(a.exports[&"color"], 5);
    }

    #[test]
    fn merging_an_empty_tally_changes_nothing() {
        let full = || {
            let mut t = Tally { files: 2, translated: 3, ..Tally::new(true) };
            t.fail("r".into(), || "e".into());
            t.infra("i".into(), || "e".into());
            *t.untraced_used.entry("k".into()).or_default() += 1;
            t
        };
        let mut t = full();
        t.merge(Tally::default());
        assert_eq!(t, full());
    }

    #[test]
    fn a_program_that_panics_leaves_no_partial_counts() {
        let mut t = Tally::default();
        t.isolate(
            || "x.pak@0x10 Ps".into(),
            |t| {
                t.translated += 1;
                t.fail("structure: y".into(), String::new);
                panic!("index out of bounds: the len is 3 but the index is 7");
            },
        );
        assert_eq!((t.translated, t.failed.len()), (0, 0));
        assert_eq!(
            t.infra,
            reasons(&[(
                "panic: index out of bounds: the len is # but the index is #",
                1,
                "x.pak@0x10 Ps"
            )])
        );
    }

    #[test]
    fn a_program_that_returns_is_merged() {
        let mut t = Tally::new(true);
        t.isolate(String::new, |t| {
            assert!(t.list, "the program's tally lists like the run's");
            t.translated += 1;
        });
        assert_eq!(t.translated, 1);
        assert!(t.infra.is_empty());
    }

    #[test]
    fn equal_panics_group_whatever_their_values() {
        let mut t = Tally::default();
        for (len, index) in [(3, 7), (12, 400)] {
            t.isolate(String::new, |_| {
                panic!("index out of bounds: the len is {len} but the index is {index}")
            });
        }
        assert_eq!(t.infra.len(), 1, "{:?}", t.infra);
        assert_eq!(t.infra.values().next().unwrap().0, 2);
    }

    #[test]
    fn panic_reasons_keep_only_the_first_line() {
        let payload: Box<dyn Any + Send> = Box::new(String::from("bad thing\nat 0x1c"));
        assert_eq!(panic_reason(&*payload), "panic: bad thing");
        let payload: Box<dyn Any + Send> = Box::new(42u32);
        assert_eq!(panic_reason(&*payload), "panic: (no message)");
    }
}

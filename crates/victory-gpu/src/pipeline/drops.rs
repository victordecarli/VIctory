//! Draws and dispatches the host did not run, counted where their cause
//! lives, in every build:
//!
//! * a shader that does not translate: on its [`Shader`], with the
//!   translator's error;
//! * a pipeline still being made, refused by the host, or made but whose
//!   draw could not be prepared: on its [`Pending`];
//! * a drop with no pipeline to blame (none set, indirect arguments the host
//!   cannot read): here, by [`Cause`].
//!
//! Counting is atomic and allocates nothing. The first drop of each shader,
//! pipeline and cause is logged once (target `victory::summary`), and
//! [`report`], with every frame report, logs the run's totals for the
//! costliest [`KEPT`] of each.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use victory_shader::ShaderError;

use super::Shader;
use super::make::Pending;

/// Items of each kind a report lists.
pub(crate) const KEPT: usize = 40;

/// Drops of one item, and whether its first was seen.
#[derive(Default)]
pub(crate) struct Count {
    drops: AtomicU64,
    seen: AtomicBool,
}

impl Count {
    /// Adds a drop; true for the item's first.
    fn add(&self) -> bool {
        self.drops.fetch_add(1, Ordering::Relaxed);
        !self.seen.swap(true, Ordering::Relaxed)
    }

    fn total(&self) -> u64 {
        self.drops.load(Ordering::Relaxed)
    }
}

/// A pipeline's drops, by why it had no host pipeline to run with.
#[derive(Default)]
pub(crate) struct PipelineDrops {
    being_made: AtomicU64,
    refused: AtomicU64,
    unprepared: AtomicU64,
    seen: AtomicBool,
}

/// Why a draw with no pipeline to blame was dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Cause {
    /// No pipeline was set.
    NoPipeline,
    /// Its pipeline has a shader `VICTORY_SKIP_PIPELINES` lists.
    #[cfg(feature = "diagnostics")]
    Skipped,
    /// Indirect: the arguments are not in a title buffer.
    IndirectArguments,
    /// Indirect: the record count is not in a title buffer.
    IndirectCount,
    /// Indirect: the host has no command signature for the kind and stride.
    IndirectSignature,
    /// Indirect: the arguments could not be bounded (no clamp, or not
    /// 4-byte aligned).
    IndirectUnbounded,
    /// Indirect: the title's signature or buffers are not ones we execute.
    IndirectLayout,
}

/// Every cause this build has, in the order reports list them.
const CAUSES: &[Cause] = &[
    Cause::NoPipeline,
    #[cfg(feature = "diagnostics")]
    Cause::Skipped,
    Cause::IndirectArguments,
    Cause::IndirectCount,
    Cause::IndirectSignature,
    Cause::IndirectUnbounded,
    Cause::IndirectLayout,
];

impl Cause {
    fn slot(self) -> usize {
        match self {
            Cause::NoPipeline => 0,
            #[cfg(feature = "diagnostics")]
            Cause::Skipped => 6,
            Cause::IndirectArguments => 1,
            Cause::IndirectCount => 2,
            Cause::IndirectSignature => 3,
            Cause::IndirectUnbounded => 4,
            Cause::IndirectLayout => 5,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Cause::NoPipeline => "no pipeline set",
            #[cfg(feature = "diagnostics")]
            Cause::Skipped => "skipped (VICTORY_SKIP_PIPELINES)",
            Cause::IndirectArguments => "indirect: arguments outside title buffers",
            Cause::IndirectCount => "indirect: count outside title buffers",
            Cause::IndirectSignature => "indirect: no host signature",
            Cause::IndirectUnbounded => "indirect: arguments not bounded",
            Cause::IndirectLayout => "indirect: signature or buffers not executed",
        }
    }
}

/// Drops by cause, at [`Cause::slot`] (one slot more than the causes of a
/// build without `diagnostics`, which never uses it).
static GENERAL: [Count; 7] =
    [const { Count { drops: AtomicU64::new(0), seen: AtomicBool::new(false) } }; 7];

/// The shaders and pipelines that dropped a draw, in the order they first did.
static SHADERS: Mutex<Vec<Arc<Shader>>> = Mutex::new(Vec::new());
static PIPELINES: Mutex<Vec<Arc<Pending>>> = Mutex::new(Vec::new());

/// The kind of a translator's error, as reports name it.
fn class(e: &ShaderError) -> &'static str {
    match e {
        ShaderError::NotDxbc => "not_dxbc",
        ShaderError::Malformed(_) => "malformed",
        ShaderError::Isa { .. } => "isa",
        ShaderError::Unsupported { .. } => "unsupported",
        ShaderError::Internal(_) => "internal",
    }
}

/// Counts a drop with no pipeline to blame.
pub(crate) fn general(cause: Cause) {
    let Some(count) = GENERAL.get(cause.slot()) else {
        return;
    };
    if count.add() {
        let frame = crate::frame::current();
        tracing::info!(target: crate::SUMMARY, cause = cause.name(), frame, "first drop");
    }
}

/// Counts a drop of a draw whose pipeline is `pending`: on its shaders that
/// do not translate, else on the pipeline.
pub(crate) fn pipeline(pending: &Arc<Pending>) {
    let (counter, cause) = match pending.made.get() {
        Some(Some(_)) => (&pending.drops.unprepared, "made, but the draw could not be prepared"),
        None => (&pending.drops.being_made, "being made"),
        Some(None) => {
            let mut blamed = false;
            for shader in &pending.shaders {
                if let Some(Err(e)) = shader.translated.get() {
                    blamed = true;
                    untranslated(shader, e);
                }
            }
            if blamed {
                return;
            }
            (&pending.drops.refused, "refused by the host")
        }
    };
    counter.fetch_add(1, Ordering::Relaxed);
    if !pending.drops.seen.swap(true, Ordering::Relaxed) {
        let frame = crate::frame::current();
        let shaders = names(&pending.shaders);
        tracing::info!(target: crate::SUMMARY, pipeline = %pending.key, %shaders, cause, frame, "first drop");
        PIPELINES.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::clone(pending));
    }
}

fn untranslated(shader: &Arc<Shader>, e: &ShaderError) {
    if shader.drops.add() {
        let frame = crate::frame::current();
        tracing::info!(
            target: crate::SUMMARY,
            stage = shader.stage,
            hash = %shader.hash,
            class = class(e),
            error = %e,
            frame,
            "first drop: the shader does not translate"
        );
        SHADERS.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::clone(shader));
    }
}

/// "stage hash" for each of `shaders`, comma-separated.
fn names(shaders: &[Arc<Shader>]) -> String {
    let names: Vec<String> = shaders.iter().map(|s| format!("{} {}", s.stage, s.hash)).collect();
    names.join(", ")
}

/// The `KEPT` items with the most drops, most first; ties by key, so the
/// same counts always give the same list.
fn costliest<T, K: Ord>(
    mut items: Vec<T>,
    total: impl Fn(&T) -> u64,
    key: impl Fn(&T) -> K,
) -> Vec<T> {
    items.sort_by(|a, b| total(b).cmp(&total(a)).then_with(|| key(a).cmp(&key(b))));
    items.truncate(KEPT);
    items
}

/// Logs the run's drop totals: the costliest shaders and pipelines, and
/// every cause with a drop. Nothing if nothing was dropped.
pub(crate) fn report() {
    let shaders = SHADERS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let pipelines = PIPELINES.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let causes: Vec<(Cause, u64)> = CAUSES
        .iter()
        .filter_map(|&c| GENERAL.get(c.slot()).map(|n| (c, n.total())))
        .filter(|&(_, n)| n > 0)
        .collect();
    if shaders.is_empty() && pipelines.is_empty() && causes.is_empty() {
        return;
    }
    let frame = crate::frame::current();
    let (shader_count, pipeline_count) = (shaders.len(), pipelines.len());
    tracing::info!(target: crate::SUMMARY, frame, shaders = shader_count, pipelines = pipeline_count, "drop totals");
    for s in costliest(shaders, |s| s.drops.total(), |s| s.hash) {
        let class = match s.translated.get() {
            Some(Err(e)) => class(e),
            Some(Ok(_)) | None => "translated",
        };
        let drops = s.drops.total();
        tracing::info!(target: crate::SUMMARY, stage = s.stage, hash = %s.hash, class, drops, "drop total: shader");
    }
    let pipeline_total = |p: &Arc<Pending>| {
        let PipelineDrops { being_made, refused, unprepared, seen: _ } = &p.drops;
        [being_made, refused, unprepared].iter().map(|n| n.load(Ordering::Relaxed)).sum()
    };
    for p in costliest(pipelines, pipeline_total, |p| p.key.clone()) {
        let PipelineDrops { being_made, refused, unprepared, seen: _ } = &p.drops;
        let [being_made, refused, unprepared] =
            [being_made, refused, unprepared].map(|n| n.load(Ordering::Relaxed));
        let shaders = names(&p.shaders);
        tracing::info!(
            target: crate::SUMMARY,
            pipeline = %p.key,
            %shaders,
            being_made,
            refused,
            unprepared,
            "drop total: pipeline"
        );
    }
    for (cause, drops) in causes {
        tracing::info!(target: crate::SUMMARY, cause = cause.name(), drops, "drop total");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_costliest_come_first_and_ties_keep_their_key_order() {
        let items = vec![("b", 5), ("a", 5), ("c", 9), ("d", 1)];
        let kept = costliest(items, |&(_, n)| n, |&(k, _)| k);
        assert_eq!(kept, [("c", 9), ("a", 5), ("b", 5), ("d", 1)]);
    }

    #[test]
    fn a_report_keeps_at_most_kept_items() {
        let items: Vec<(usize, u64)> = (0..KEPT + 5).map(|i| (i, 1)).collect();
        let kept = costliest(items, |&(_, n)| n, |&(k, _)| k);
        assert_eq!(kept.len(), KEPT);
        assert_eq!(kept.first(), Some(&(0, 1)));
    }

    #[test]
    fn every_cause_has_its_own_slot() {
        let mut slots: Vec<usize> = CAUSES.iter().map(|c| c.slot()).collect();
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(slots.len(), CAUSES.len());
        assert!(slots.iter().all(|&s| s < GENERAL.len()));
    }

    fn pending(shaders: &[&Arc<Shader>]) -> Arc<Pending> {
        Arc::new(Pending::new(shaders.iter().map(|&s| Arc::clone(s)).collect(), None))
    }

    #[test]
    fn a_shader_that_does_not_translate_takes_its_pipelines_drops() {
        let bad = super::super::shader("pixel", b"drops: not a container");
        assert!(bad.translated().is_none());
        let p = pending(&[&bad]);
        // What a worker leaves when a shader does not translate.
        p.made.get_or_init(|| None);
        pipeline(&p);
        pipeline(&p);
        assert_eq!(bad.drops.total(), 2);
        assert_eq!(p.drops.refused.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_pipeline_refused_with_no_failed_shader_takes_the_drop() {
        let s = super::super::shader("vertex", b"drops: never translated");
        let p = pending(&[&s]);
        p.made.get_or_init(|| None);
        pipeline(&p);
        assert_eq!(p.drops.refused.load(Ordering::Relaxed), 1);
        assert_eq!(s.drops.total(), 0);
    }

    #[test]
    fn a_pipeline_not_made_yet_counts_as_being_made() {
        let s = super::super::shader("compute", b"drops: being made");
        let p = pending(&[&s]);
        pipeline(&p);
        assert_eq!(p.drops.being_made.load(Ordering::Relaxed), 1);
        assert_eq!(p.drops.refused.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn only_the_first_drop_of_an_item_is_its_first() {
        let count = Count::default();
        assert!(count.add());
        assert!(!count.add());
        assert_eq!(count.total(), 2);
    }
}

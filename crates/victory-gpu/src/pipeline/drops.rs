//! Draws and dispatches the host did not run, counted where their cause
//! lives, in every build:
//!
//! * a shader that does not translate: on its [`Shader`], with the
//!   translator's error;
//! * a pipeline still being made or refused by the host: on its
//!   [`Pending`];
//! * a made pipeline whose draw could not be prepared: on its [`Pending`]
//!   by [`Unprepared`] reason, and on that reason's [`Cause`];
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

/// A pipeline's drops, by why it had no host pipeline to run with, or why
/// its draw could not be prepared.
#[derive(Default)]
pub(crate) struct PipelineDrops {
    being_made: AtomicU64,
    refused: AtomicU64,
    unprepared: Reasons,
    seen: AtomicBool,
}

/// Why a draw, mesh dispatch, dispatch or indirect execution did not run on
/// the host.
#[derive(Clone, Copy, Debug)]
pub(crate) enum NotRun {
    /// No host pipeline to run with: none set, not made yet, refused, a
    /// shader that does not translate, or skipped (`count_drop` tells which).
    NoHostPipeline,
    /// No pipeline to blame.
    General(Cause),
    /// Its pipeline is made, but the draw could not be prepared.
    Unprepared(Unprepared),
}

/// Why the draw of a made pipeline could not be prepared on the host, with
/// what its first case is logged with.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code, reason = "the context fields are read through `Debug`, by the first drop line")]
pub(crate) enum Unprepared {
    /// A draw with a compute or mesh pipeline, or a mesh dispatch with
    /// another (`mesh` is what was asked for).
    OtherKind { mesh: bool },
    /// No dynamic descriptor slots for a fresh table of `n` descriptors.
    NoDynamicSlots { n: u32 },
    /// The index buffer is not in a title buffer.
    IndexBuffer { address: u64, size: u32 },
    /// Rectangle vertices outside their buffer (`size`, its bytes).
    RectangleVertices { slot: u32, address: u64, size: u32 },
    /// No scratch memory for the expanded rectangles.
    Scratch { bytes: u64 },
    /// The host list cannot record mesh dispatches (`code`, the
    /// `HRESULT` of the cast).
    MeshList { code: i32 },
    /// Rectangles drawn indexed, which are not expanded.
    IndexedRectangles,
    /// Rectangles drawn indirectly, which are not expanded.
    IndirectRectangles,
}

impl Unprepared {
    pub(crate) fn cause(self) -> Cause {
        match self {
            Unprepared::OtherKind { .. } => Cause::OtherKind,
            Unprepared::NoDynamicSlots { .. } => Cause::NoDynamicSlots,
            Unprepared::IndexBuffer { .. } => Cause::IndexBuffer,
            Unprepared::RectangleVertices { .. } => Cause::RectangleVertices,
            Unprepared::Scratch { .. } => Cause::Scratch,
            Unprepared::MeshList { .. } => Cause::MeshList,
            Unprepared::IndexedRectangles => Cause::IndexedRectangles,
            Unprepared::IndirectRectangles => Cause::IndirectRectangles,
        }
    }

    /// Its field in a `drop total: pipeline` line.
    #[cfg(test)]
    fn field(self) -> &'static str {
        match self {
            Unprepared::OtherKind { .. } => "other_kind",
            Unprepared::NoDynamicSlots { .. } => "no_dynamic_slots",
            Unprepared::IndexBuffer { .. } => "index_buffer",
            Unprepared::RectangleVertices { .. } => "rectangle_vertices",
            Unprepared::Scratch { .. } => "scratch",
            Unprepared::MeshList { .. } => "mesh_list",
            Unprepared::IndexedRectangles => "indexed_rectangles",
            Unprepared::IndirectRectangles => "indirect_rectangles",
        }
    }
}

/// A made pipeline's drops by [`Unprepared`] reason.
#[derive(Default)]
struct Reasons {
    other_kind: AtomicU64,
    no_dynamic_slots: AtomicU64,
    index_buffer: AtomicU64,
    rectangle_vertices: AtomicU64,
    scratch: AtomicU64,
    mesh_list: AtomicU64,
    indexed_rectangles: AtomicU64,
    indirect_rectangles: AtomicU64,
}

impl Reasons {
    fn add(&self, reason: Unprepared) {
        let counter = match reason {
            Unprepared::OtherKind { .. } => &self.other_kind,
            Unprepared::NoDynamicSlots { .. } => &self.no_dynamic_slots,
            Unprepared::IndexBuffer { .. } => &self.index_buffer,
            Unprepared::RectangleVertices { .. } => &self.rectangle_vertices,
            Unprepared::Scratch { .. } => &self.scratch,
            Unprepared::MeshList { .. } => &self.mesh_list,
            Unprepared::IndexedRectangles => &self.indexed_rectangles,
            Unprepared::IndirectRectangles => &self.indirect_rectangles,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Each reason's field name and count, in the order lines list them.
    fn counts(&self) -> [(&'static str, u64); 8] {
        let Reasons {
            other_kind,
            no_dynamic_slots,
            index_buffer,
            rectangle_vertices,
            scratch,
            mesh_list,
            indexed_rectangles,
            indirect_rectangles,
        } = self;
        [
            ("other_kind", other_kind),
            ("no_dynamic_slots", no_dynamic_slots),
            ("index_buffer", index_buffer),
            ("rectangle_vertices", rectangle_vertices),
            ("scratch", scratch),
            ("mesh_list", mesh_list),
            ("indexed_rectangles", indexed_rectangles),
            ("indirect_rectangles", indirect_rectangles),
        ]
        .map(|(name, n)| (name, n.load(Ordering::Relaxed)))
    }

    fn total(&self) -> u64 {
        self.counts().iter().map(|&(_, n)| n).sum()
    }
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
    /// Indirect: the title's signature (or its absence) or buffers are not
    /// ones we execute.
    IndirectLayout,
    /// A dispatch whose pipeline is a graphics pipeline.
    ComputeKind,
    /// A mesh dispatch whose group count is in GPU memory.
    IndirectMesh,
    /// The [`Unprepared`] reasons, one each.
    OtherKind,
    NoDynamicSlots,
    IndexBuffer,
    RectangleVertices,
    Scratch,
    MeshList,
    IndexedRectangles,
    IndirectRectangles,
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
    Cause::ComputeKind,
    Cause::IndirectMesh,
    Cause::OtherKind,
    Cause::NoDynamicSlots,
    Cause::IndexBuffer,
    Cause::RectangleVertices,
    Cause::Scratch,
    Cause::MeshList,
    Cause::IndexedRectangles,
    Cause::IndirectRectangles,
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
            Cause::ComputeKind => 7,
            Cause::IndirectMesh => 8,
            Cause::OtherKind => 9,
            Cause::NoDynamicSlots => 10,
            Cause::IndexBuffer => 11,
            Cause::RectangleVertices => 12,
            Cause::Scratch => 13,
            Cause::MeshList => 14,
            Cause::IndexedRectangles => 15,
            Cause::IndirectRectangles => 16,
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
            Cause::ComputeKind => "dispatch with a graphics pipeline",
            Cause::IndirectMesh => "indirect mesh dispatch",
            Cause::OtherKind => "draw or mesh dispatch with another kind of pipeline",
            Cause::NoDynamicSlots => "no dynamic descriptor slots for a table",
            Cause::IndexBuffer => "index buffer outside title buffers",
            Cause::RectangleVertices => "rectangle vertices outside their buffer",
            Cause::Scratch => "no scratch memory for rectangles",
            Cause::MeshList => "host list cannot record mesh dispatches",
            Cause::IndexedRectangles => "indexed rectangles",
            Cause::IndirectRectangles => "indirect rectangles",
        }
    }
}

/// Drops by cause, at [`Cause::slot`] (one slot more than the causes of a
/// build without `diagnostics`, which never uses it).
static GENERAL: [Count; 17] =
    [const { Count { drops: AtomicU64::new(0), seen: AtomicBool::new(false) } }; 17];

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

/// Counts a drop of a draw whose pipeline is `pending` and could not be
/// prepared for `reason`: on the pipeline by reason and on the reason's
/// cause, the first with its context. A pipeline not made (yet) is to
/// blame instead, as [`pipeline`] counts it.
pub(crate) fn unprepared(pending: &Arc<Pending>, reason: Unprepared) {
    let Some(Some(_)) = pending.made.get() else {
        pipeline(pending);
        return;
    };
    pending.drops.unprepared.add(reason);
    first_of_pipeline(pending, "made, but the draw could not be prepared");
    let cause = reason.cause();
    if GENERAL.get(cause.slot()).is_some_and(Count::add) {
        let frame = crate::frame::current();
        tracing::info!(target: crate::SUMMARY, cause = cause.name(), detail = ?reason, frame, "first drop");
    }
}

/// Counts a drop of a draw whose pipeline is `pending` and has no host
/// pipeline to run with: on its shaders that do not translate, else on the
/// pipeline.
pub(crate) fn pipeline(pending: &Arc<Pending>) {
    let (counter, cause) = match pending.made.get() {
        // Made since the draw found none: it was still being made then.
        None | Some(Some(_)) => (&pending.drops.being_made, "being made"),
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
    first_of_pipeline(pending, cause);
}

/// Logs the first drop of `pending`, and keeps it for the reports.
fn first_of_pipeline(pending: &Arc<Pending>, cause: &'static str) {
    if !pending.drops.seen.swap(true, Ordering::Relaxed) {
        let frame = crate::frame::current();
        let shaders = names(&pending.shaders);
        tracing::info!(target: crate::SUMMARY, pipeline = %pending.key, %shaders, cause, frame, "first drop");
        PIPELINES.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::clone(pending));
    }
}

/// The `drop total: pipeline` line of `p`: its totals, then the count of
/// each reason its draws could not be prepared for, only those it has (a
/// field that is `None` is not written).
fn pipeline_line(p: &Pending) {
    let PipelineDrops { being_made, refused, unprepared: reasons, seen: _ } = &p.drops;
    let [being_made, refused] = [being_made, refused].map(|n| n.load(Ordering::Relaxed));
    let unprepared = reasons.total();
    let [
        other_kind,
        no_dynamic_slots,
        index_buffer,
        rectangle_vertices,
        scratch,
        mesh_list,
        indexed_rectangles,
        indirect_rectangles,
    ] = reasons.counts().map(|(_, n)| (n > 0).then_some(n));
    let shaders = names(&p.shaders);
    tracing::info!(
        target: crate::SUMMARY,
        pipeline = %p.key,
        %shaders,
        being_made,
        refused,
        unprepared,
        other_kind,
        no_dynamic_slots,
        index_buffer,
        rectangle_vertices,
        scratch,
        mesh_list,
        indexed_rectangles,
        indirect_rectangles,
        "drop total: pipeline"
    );
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
        [being_made, refused].iter().map(|n| n.load(Ordering::Relaxed)).sum::<u64>()
            + unprepared.total()
    };
    for p in costliest(pipelines, pipeline_total, |p| p.key.clone()) {
        pipeline_line(&p);
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
    fn the_dispatch_causes_are_listed_with_names_of_their_own() {
        for cause in [Cause::ComputeKind, Cause::IndirectMesh] {
            assert!(CAUSES.contains(&cause));
            assert_eq!(CAUSES.iter().filter(|c| c.name() == cause.name()).count(), 1);
        }
    }

    #[test]
    fn a_general_drop_of_a_dispatch_cause_is_counted() {
        let count = |cause: Cause| GENERAL.get(cause.slot()).map_or(0, Count::total);
        let (kind, mesh) = (count(Cause::ComputeKind), count(Cause::IndirectMesh));
        general(Cause::ComputeKind);
        general(Cause::IndirectMesh);
        general(Cause::IndirectMesh);
        assert_eq!(count(Cause::ComputeKind), kind + 1);
        assert_eq!(count(Cause::IndirectMesh), mesh + 2);
    }

    /// One of each reason, with made-up context.
    const EVERY_REASON: [Unprepared; 8] = [
        Unprepared::OtherKind { mesh: true },
        Unprepared::NoDynamicSlots { n: 64 },
        Unprepared::IndexBuffer { address: 0x1000, size: 96 },
        Unprepared::RectangleVertices { slot: 2, address: 0x2000, size: 48 },
        Unprepared::Scratch { bytes: 4096 },
        Unprepared::MeshList { code: -2147467262 },
        Unprepared::IndexedRectangles,
        Unprepared::IndirectRectangles,
    ];

    #[test]
    fn each_reason_counts_on_its_own_named_counter_and_they_sum_to_unprepared() {
        let reasons = Reasons::default();
        // Reason k counted k + 1 times: a counter shared by two reasons
        // would hold a sum no single k gives.
        for (k, &reason) in EVERY_REASON.iter().enumerate() {
            for _ in 0..=k {
                reasons.add(reason);
            }
        }
        let counts = reasons.counts();
        for (k, &reason) in EVERY_REASON.iter().enumerate() {
            let named = counts.iter().find(|(name, _)| *name == reason.field());
            assert_eq!(named.map(|&(_, n)| n), Some(k as u64 + 1), "{reason:?}");
        }
        assert_eq!(reasons.total(), (1..=8).sum::<u64>());
    }

    #[test]
    fn a_pipeline_line_names_only_the_reasons_it_has() {
        let reasons = Reasons::default();
        reasons.add(Unprepared::NoDynamicSlots { n: 8 });
        reasons.add(Unprepared::NoDynamicSlots { n: 16 });
        reasons.add(Unprepared::Scratch { bytes: 1 });
        let present: Vec<(&str, u64)> =
            reasons.counts().into_iter().filter(|&(_, n)| n > 0).collect();
        assert_eq!(present, [("no_dynamic_slots", 2), ("scratch", 1)]);
        assert_eq!(reasons.total(), 3);
    }

    #[test]
    fn every_reason_has_a_cause_of_its_own() {
        let mut causes: Vec<Cause> = EVERY_REASON.iter().map(|r| r.cause()).collect();
        causes.dedup();
        assert_eq!(causes.len(), EVERY_REASON.len());
        assert!(causes.iter().all(|c| CAUSES.contains(c)));
    }

    #[test]
    fn only_the_first_drop_of_an_item_is_its_first() {
        let count = Count::default();
        assert!(count.add());
        assert!(!count.add());
        assert_eq!(count.total(), 2);
    }
}

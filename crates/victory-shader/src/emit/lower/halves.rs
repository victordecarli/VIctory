//! Console wave64 compute programs on a host whose waves have 32 lanes
//! (NVIDIA: `WaveLaneCountMax` 32), which refuses `WaveSize(64)`. One
//! console wave runs as two host waves of 32, its halves: console lanes
//! 0..31 and 32..63. Code that reads other lanes gets console lane numbers,
//! and what lies in the other half (a lane mask read as a number, a lane
//! read, the wave's first lane) the two halves exchange, one dword each,
//! through LDS at a group barrier.
//!
//! The split is exact only where the halves cannot part. A group barrier,
//! the exchange's included, waits for every thread of the group, and a
//! scalar register has a copy per host wave where the console has one per
//! wave, so a block that only one half enters would leave the other waiting
//! or with a stale scalar. A program is split only when ([`keeps_64`]):
//! - its thread group is one console wave (64 threads), or at most 32
//!   threads (a single host wave, which exchanges nothing);
//! - in a group of two host waves, the program's own LDS is accessed only
//!   in blocks every lane runs: the console compiler leaves out the
//!   barriers one wave sharing LDS needs, so a group barrier before each
//!   access keeps the console wave's order across the halves;
//! - in a group of two host waves, its whole-wave sections
//!   ([`super::super::wave`]) lie in blocks every lane runs, where the halves
//!   exchange their results, and sum no floats (the console adds them in an
//!   order of its own); it has no `ds_(b)permute`;
//! - in a group of two host waves, every place the halves exchange lies in
//!   a block every lane runs;
//! - a block only some lanes run holds no group barrier, and a scalar
//!   written there (other than a lane mask) is read only past its write:
//!   not through a phi at a join, nor where every lane runs. A half that
//!   skipped the block keeps an older copy, which it then never reads;
//!
//! A block in a loop each lane leaves on its own is not one every lane
//! runs, the loop's closing block included: the halves could go round a
//! different number of times.
//!
//! Where a block only some lanes run is what keeps a program at 64 lanes,
//! the lane branch around it can be taken as the console takes it, one
//! decision for the whole wave ([`widened`]): each half asks its lanes, the
//! halves exchange the answer ([`Lower::wave_wide`]) and go the same way,
//! and EXEC stays a guard on the vector instructions inside
//! ([`crate::structure::structure_wide`]). Every lane then runs the branch's
//! blocks. A branch closing a loop each lane leaves on its own is taken so
//! too: the wave goes round again while some lane of it is left, and the
//! lanes that left stay out of the vector instructions. Only branches in no
//! other lane branch's region are taken so, outermost first.
//!
//! Every other console wave64 program keeps 64-lane waves.
//!
//! Ported from the fxceless line (`base/fxceless`, `emit/lower.rs`:
//! `wave_base`, `console_thread`, `console_lane`, `exchange_pair`,
//! `exchange_halves`, `branch_regions`, `all_lane_blocks`, `wave_wide`, and
//! `emit/mod.rs`: `widen`), restricted to the cases above.

use super::super::wave::{Op as WaveOp, Shape};
use super::*;

/// LDS dwords a split program adds past its own: the counter numbering
/// the group's two host waves, then the exchange pair. None for a group of
/// a single host wave. Values made once, in the entry block.
pub(super) struct Halves {
    two: Option<(u32, u32)>,
    /// Two host waves sharing the program's own LDS: a group barrier
    /// before each access.
    lds: bool,
    base: Option<V>,
    lane: Option<V>,
}

impl Halves {
    /// The halves of a group of `group` threads, taking their LDS dwords
    /// past `lds_dwords`.
    pub(super) fn new(lds_dwords: &mut u32, group: u32) -> Self {
        let lds = group > 32 && *lds_dwords > 0;
        let two = (group > 32).then(|| {
            let counter = *lds_dwords;
            *lds_dwords += DWORDS;
            (counter, counter + 1)
        });
        Halves { two, lds, base: None, lane: None }
    }

    /// Whether each access to the program's own LDS needs a group barrier
    /// before it.
    pub(super) fn orders_lds(&self) -> bool {
        self.lds
    }

    /// The LDS dwords these halves add past the program's own.
    pub(super) fn dwords(&self) -> u32 {
        if self.two.is_some() { DWORDS } else { 0 }
    }
}

/// The counter and the exchange pair.
const DWORDS: u32 = 3;

/// `first`, the lowering of `x`, or where it keeps a console wave64 compute
/// program at 64-lane waves on a host of 32-lane waves, the lowering with
/// the lane branches [`to_widen`] names taken wave-wide
/// ([`crate::structure::structure_wide`]), widening on until the program
/// runs in halves. Where it never does, or a wider lowering fails, `first`.
pub(super) fn widened(x: &Parts, first: Lowered) -> Lowered {
    let StageIo::Cs(cs) = x.native.io else { return first };
    if first.wave_size != Some(64) || x.host_lanes != 32 || !x.native.wave64 {
        return first;
    }
    let group = cs.threads.iter().product::<u32>();
    let lds_dwords = cs.lds_bytes() / 4;
    let mut wide = HashSet::new();
    let mut st = None;
    // (each round takes at least one more branch wave-wide)
    for _ in 0..x.p.blocks.len() {
        let more = to_widen(x.p, st.as_ref().unwrap_or(x.st), group, lds_dwords);
        if more.is_empty() {
            break;
        }
        wide.extend(more);
        let Ok(now) = crate::structure::structure_wide(x.p, &wide) else { break };
        let Ok(res) = crate::resource::track(x.p, &x.native.layout, x.native.stage, Some(&now))
        else {
            break;
        };
        let parts = Parts {
            native: x.native,
            p: x.p,
            st: &now,
            io: x.io,
            res: &res,
            inputs: x.inputs.clone(),
            outputs: x.outputs.clone(),
            prims: x.prims.clone(),
            mesh_stride: x.mesh_stride,
            host_lanes: x.host_lanes,
        };
        match lower_once(&parts) {
            Ok(l) if l.wave_size == Some(32) => return l,
            Ok(_) => st = Some(now),
            Err(_) => break,
        }
    }
    first
}

/// Why the console wave64 compute program `p` (`group` threads, `lds_dwords`
/// of LDS) keeps 64-lane waves on a host of 32-lane waves; None when it runs
/// in halves.
pub(super) fn keeps_64(p: &Program, st: &Structured, group: u32, lds_dwords: u32) -> Option<Why> {
    keeps_64_at(p, st, group, lds_dwords).map(|k| k.why)
}

/// What keeps a program at 64-lane waves.
struct Keep {
    why: Why,
    /// The block, one not every lane runs, that holds it: a lane branch
    /// around it taken wave-wide ([`to_widen`]) would have every lane run it.
    at: Option<usize>,
}

impl Keep {
    fn fixed(why: Why) -> Option<Keep> {
        Some(Keep { why, at: None })
    }

    fn at(why: Why, b: usize) -> Option<Keep> {
        Some(Keep { why, at: Some(b) })
    }
}

/// [`keeps_64`], with where the reason lies.
fn keeps_64_at(p: &Program, st: &Structured, group: u32, lds_dwords: u32) -> Option<Keep> {
    let two = match group {
        0..=32 => false,
        64 => true,
        _ => return Keep::fixed("a thread group of other than one console wave"),
    };
    let Some(all) = all_lane_blocks(st) else {
        return Keep::fixed("an indirect exit");
    };
    let every = |b: usize| all.get(b).copied().unwrap_or(false);
    if let Some(keep) = sections_keep_64(p, two, every) {
        return Some(keep);
    }
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let written: HashMap<Value, usize> = p
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(b, block)| {
            let insts =
                block.insts.iter().flat_map(move |i| i.writes.iter().map(move |w| (w.1, b)));
            insts.chain(block.phis.iter().map(move |phi| (phi.value, b)))
        })
        .collect();
    // Scalars written where not every lane runs, with their block.
    let mut lane_scalars = HashMap::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for inst in &block.insts {
            let n = inst.name();
            if n.starts_with("ds_permute") || n.starts_with("ds_bpermute") {
                return Keep::fixed("ds_(b)permute");
            }
            if two && !every(b) && matches!(n, "v_readlane_b32" | "v_readfirstlane_b32") {
                return Keep::at("a lane read where not every lane runs", b);
            }
            if two && reads_mask_as_data(st, inst) {
                // The ballot is taken where the mask is written (`mask_dword`).
                let mut mask_blocks = mask_operands(inst).map(|v| {
                    let low = st.highs.get(&v).copied().unwrap_or(v);
                    // (program inputs, EXEC on entry, are the entry block's)
                    written.get(&low).copied().unwrap_or(0)
                });
                if let Some(m) = mask_blocks.find(|&m| !every(m)) {
                    const WHY: Why =
                        "a lane mask read as a number, written where not every lane runs";
                    return Keep::at(WHY, m);
                }
            }
            if every(b) {
                continue;
            }
            if two && lds_dwords > 0 && lds_access(inst) {
                return Keep::at("LDS accessed where not every lane runs", b);
            }
            if n == "s_barrier" {
                return Keep::at("a group barrier where not every lane runs", b);
            }
            let mask_op = inst.writes.iter().any(|&(r, w)| match r {
                Reg::S(c) => c == EXEC || c == EXEC + 1 || is_mask(w),
                Reg::Scc | Reg::V(_) => false,
            });
            for &(r, w) in &inst.writes {
                let scalar = match r {
                    Reg::S(c) => !(c == EXEC || c == EXEC + 1 || is_mask(w)),
                    // A mask operation's SCC (some lane left) is lowered
                    // only as a branch condition, which takes it per lane.
                    Reg::Scc => !mask_op,
                    Reg::V(_) => false,
                };
                if scalar {
                    lane_scalars.insert(w, b);
                }
            }
        }
    }
    lane_scalar_read_past(p, &lane_scalars, every)
}

/// A scalar of `lane_scalars` (written where not every lane runs, in the
/// block given) read past its write. Read past the write only (dominated by
/// it, as SSA has every read but a phi's), the copy of a half that skipped
/// the block is never read. Through a phi at a join, or where every lane
/// runs, it would be, where the console's wave has the new value.
fn lane_scalar_read_past(
    p: &Program,
    lane_scalars: &HashMap<Value, usize>,
    every: impl Fn(usize) -> bool,
) -> Option<Keep> {
    let phi_reads = p.blocks.iter().flat_map(|block| &block.phis).flat_map(|phi| &phi.args);
    let all_lane_reads = p
        .blocks
        .iter()
        .enumerate()
        .filter(|&(b, _)| every(b))
        .flat_map(|(_, block)| &block.insts)
        .flat_map(|i| i.reads.iter().map(|&(_, v)| v));
    let written = phi_reads.copied().chain(all_lane_reads).find_map(|v| lane_scalars.get(&v));
    written.and_then(|&b| Keep::at("a scalar written where not every lane runs, read past it", b))
}

/// The lane branches to take wave-wide next so that `p` could run in
/// halves: those around the block holding the reason [`keeps_64`] gives
/// that lie in no other lane branch's region (a branch closing a loop each
/// lane leaves on its own lies in its own). Empty when there is none.
pub(super) fn to_widen(p: &Program, st: &Structured, group: u32, lds_dwords: u32) -> Vec<usize> {
    let Some(Keep { at: Some(at), .. }) = keeps_64_at(p, st, group, lds_dwords) else {
        return Vec::new();
    };
    let Some(regions) = lane_regions(st) else {
        return Vec::new();
    };
    let inside_another = |d: usize| {
        regions.iter().any(|r| r.branch != d && r.blocks.get(d).copied().unwrap_or(false))
    };
    regions
        .iter()
        .filter(|r| !inside_another(r.branch) && r.blocks.get(at).copied().unwrap_or(false))
        .map(|r| r.branch)
        .collect()
}

/// Why the whole-wave sections of `p` keep 64-lane waves. In a group of two
/// host waves each section lies where every lane runs, so that both halves
/// reach the exchange of their results ([`Lower::whole_wave`]), and sums no
/// floats: the console adds them in its own order.
fn sections_keep_64(p: &Program, two: bool, every: impl Fn(usize) -> bool) -> Option<Keep> {
    if !two {
        return None;
    }
    for b in 0..p.blocks.len() {
        for section in super::super::wave::sections(p, b).values() {
            if !every(b) {
                return Keep::at("a whole-wave section where not every lane runs", b);
            }
            let float_sum = section.shapes.values().any(|s| {
                matches!(s, Shape::Active(WaveOp::AddF, ..) | Shape::Prefix(WaveOp::AddF, ..))
            });
            if float_sum {
                return Keep::fixed(FLOAT_SUM);
            }
        }
    }
    None
}

const FLOAT_SUM: Why = "a float wave sum over two host waves";

/// An access to the program's LDS: not `ds_swizzle` or `ds_(b)permute`,
/// which move data between lanes without it.
pub(super) fn lds_access(inst: &Inst) -> bool {
    let n = inst.name();
    matches!(inst.fields, Fields::Ds { .. })
        && !(n.starts_with("ds_swizzle") || n.contains("permute"))
}

/// The values of the scalar operands of `inst`.
fn mask_operands(inst: &Inst) -> impl Iterator<Item = Value> + '_ {
    let ops: Vec<&Operand> = match &inst.fields {
        Fields::Sop { src, .. } => src.iter().collect(),
        Fields::Valu(v) => v.src.iter().collect(),
        _ => Vec::new(),
    };
    ops.into_iter().filter_map(move |op| match *op {
        Operand::Scalar { code, .. } if code != NULL => inst.read(Reg::S(code)),
        _ => None,
    })
}

/// Per IR block: every lane of the wave that started the program runs it,
/// as no per-lane branch separates it from the entry (it lies in no lane
/// branch's region, [`lane_regions`]). A branch the wave takes together
/// ([`Structured::wide`]) separates no lanes. None with an indirect exit,
/// which is not followed.
pub(super) fn all_lane_blocks(st: &Structured) -> Option<Vec<bool>> {
    let mut all = vec![true; st.exits.len()];
    for r in lane_regions(st)? {
        for (a, inside) in all.iter_mut().zip(r.blocks) {
            *a &= !inside;
        }
    }
    Some(all)
}

/// The blocks some lanes of a wave may run and others not, past a per-lane
/// branch.
struct Region {
    /// The block the branch closes.
    branch: usize,
    /// Per IR block: reachable from the branch without post-dominating it,
    /// and in a loop the branch's own block.
    blocks: Vec<bool>,
}

/// The region of every per-lane branch that is not wide. None with an
/// indirect exit, which is not followed.
fn lane_regions(st: &Structured) -> Option<Vec<Region>> {
    let n = st.exits.len();
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, e) in st.exits.iter().enumerate() {
        match *e {
            LaneExit::Jump(t) => succ[b].push(t),
            LaneExit::Branch { taken, not_taken, .. } => succ[b].extend([taken, not_taken]),
            LaneExit::End => {}
            LaneExit::Indirect => return None,
        }
    }
    let pdom = post_dominators(&succ);
    let mut regions = Vec::new();
    for (d, e) in st.exits.iter().enumerate() {
        let LaneExit::Branch { when, .. } = *e else { continue };
        if st.wide.contains(&d)
            || st.bools.support(when).iter().all(|a| matches!(a, Atom::Uniform(_)))
        {
            continue; // every lane goes the same way
        }
        let mut blocks = vec![false; n];
        let mut cycle = false;
        let mut stack: Vec<usize> = succ[d].clone();
        while let Some(b) = stack.pop() {
            cycle |= b == d;
            if blocks[b] || pdom[d][b] {
                continue;
            }
            blocks[b] = true;
            stack.extend(&succ[b]);
        }
        // In a loop each lane leaves on its own, the branch's block too runs
        // as often as each lane goes round (it post-dominates itself).
        blocks[d] |= cycle;
        regions.push(Region { branch: d, blocks });
    }
    Some(regions)
}

/// `pdom[b][c]`: block c post-dominates block b (every path from b to the
/// end passes through c; a block post-dominates itself).
fn post_dominators(succ: &[Vec<usize>]) -> Vec<Vec<bool>> {
    let n = succ.len();
    let mut pdom: Vec<Vec<bool>> = (0..n)
        .map(|b| if succ[b].is_empty() { (0..n).map(|c| c == b).collect() } else { vec![true; n] })
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            if succ[b].is_empty() {
                continue;
            }
            let mut v: Vec<bool> = (0..n).map(|c| succ[b].iter().all(|&t| pdom[t][c])).collect();
            v[b] = true;
            if v != pdom[b] {
                pdom[b] = v;
                changed = true;
            }
        }
    }
    pdom
}

impl Lower<'_> {
    /// The first console thread of this host wave (0 or 32) in a group of two
    /// host waves. A GPU need not give a host wave consecutive thread indices,
    /// so each wave takes the next 32 from an LDS counter: every thread zeroes
    /// it, the group meets, each wave's first lane adds 32 and the wave takes
    /// what it found. In the entry block, which every thread runs.
    fn wave_base(&mut self) -> Option<V> {
        let (counter, _) = self.halves.as_ref()?.two?;
        if let Some(v) = self.halves.as_ref().and_then(|h| h.base) {
            return Some(v);
        }
        let (e, i32t) = (self.s.entry, self.s.t.i32);
        let lds = self.s.lds();
        let k = self.s.m.i32(counter);
        let ptr = self.s.m.element(e, lds, k);
        let zero = self.s.m.i32(0);
        self.s.m.store(e, ptr, zero);
        self.group_sync(e);
        let lane = self.s.op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let first = self.s.m.cmp(e, pred::EQ, lane, zero);
        let k32 = self.s.m.i32(32);
        let add = self.s.m.select(e, first, k32, zero);
        let ptr = self.s.m.element(e, lds, k);
        let old = self.s.m.atomic(e, super::super::bitcode::Rmw::Add, ptr, add);
        let base = self.s.op(e, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[old]);
        self.s.flags |= flags::WAVE_OPS;
        if let Some(h) = self.halves.as_mut() {
            h.base = Some(base);
        }
        Some(base)
    }

    /// This thread's console thread index in the group: [`Self::wave_base`]
    /// plus the host lane; in a group of a single host wave, the host lane.
    pub(super) fn console_thread(&mut self) -> V {
        let (e, i32t) = (self.s.entry, self.s.t.i32);
        let lane = self.s.op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        match self.wave_base() {
            Some(base) => self.s.m.bin(e, Bin::Add, base, lane),
            None => lane,
        }
    }

    /// This thread's console lane (0..63).
    pub(super) fn console_lane(&mut self) -> V {
        if let Some(v) = self.halves.as_ref().and_then(|h| h.lane) {
            return v;
        }
        let e = self.s.entry;
        let t = self.console_thread();
        let k63 = self.s.m.i32(63);
        let lane = self.s.m.bin(e, Bin::And, t, k63);
        self.s.flags |= flags::WAVE_OPS;
        if let Some(h) = self.halves.as_mut() {
            h.lane = Some(lane);
        }
        lane
    }

    /// Each half gives `mine` (the same in all its lanes); both get (the
    /// lower half's, the upper half's). A group of a single host wave has no
    /// upper half: 0 for it. At `at`, which every thread of the group runs.
    pub(super) fn exchange(&mut self, at: usize, mine: V) -> (V, V) {
        let Some((_, pair)) = self.halves.as_ref().and_then(|h| h.two) else {
            return (mine, self.s.m.i32(0));
        };
        let lane = self.console_lane();
        let five = self.s.m.i32(5);
        let half = self.s.m.bin(at, Bin::LShr, lane, five);
        let first = self.s.m.i32(pair);
        let index = self.s.m.bin(at, Bin::Add, first, half);
        let lds = self.s.lds();
        let ptr = self.s.m.element(at, lds, index);
        self.s.m.store(at, ptr, mine);
        self.group_sync(at);
        let second = self.s.m.i32(pair + 1);
        let p0 = self.s.m.element(at, lds, first);
        let p1 = self.s.m.element(at, lds, second);
        let lo = self.s.m.load(at, p0);
        let hi = self.s.m.load(at, p1);
        // The pair is written again only after every thread has read it.
        self.group_sync(at);
        (lo, hi)
    }

    /// `v_readlane` of console lane `lane` (uniform): the half holding it
    /// reads it, and the halves exchange.
    pub(super) fn console_read_lane(&mut self, x: V, lane: V) -> V {
        let c = self.cur;
        let mine = self.read_lane(x, lane);
        let (lo, hi) = self.exchange(c, mine);
        let (five, z) = (self.s.m.i32(5), self.s.m.i32(0));
        let half = self.s.m.bin(c, Bin::LShr, lane, five);
        let lower = self.s.m.cmp(c, pred::EQ, half, z);
        self.s.m.select(c, lower, lo, hi)
    }

    /// `v_readfirstlane`: the console wave's first lane in EXEC. Where every
    /// lane runs (as [`keeps_64`] requires of a group of two host waves),
    /// that is lane 0, in the lower half, unless EXEC is narrowed there to
    /// `guard` (without a branch, or inside a branch taken wave-wide). Then
    /// each half finds its first lane in EXEC, the halves exchange whether
    /// they have one and its value, and the lower half's is taken if it has
    /// one. With no lane in EXEC, lane 0's, as the ISA reads then.
    pub(super) fn console_first_lane(&mut self, x: V, guard: Option<V>) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        self.s.flags |= flags::WAVE_OPS;
        let Some(on) = guard else {
            let mine = self.s.op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[x]);
            return self.exchange(c, mine).0;
        };
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let ballot = self.s.op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[on]);
        let ballot = self.s.m.extract(c, ballot, 0);
        // DXIL's FirstbitLo: -1 for none.
        let first = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[ballot]);
        let zero = self.s.m.i32(0);
        let none = self.s.m.cmp(c, pred::EQ, ballot, zero);
        let lane = self.s.m.select(c, none, zero, first);
        let mine = self.read_lane(x, lane);
        let (lower_lanes, upper_lanes) = self.exchange(c, ballot);
        let (lower, upper) = self.exchange(c, mine);
        let lower_has = self.s.m.cmp(c, pred::NE, lower_lanes, zero);
        let upper_has = self.s.m.cmp(c, pred::NE, upper_lanes, zero);
        let other = self.s.m.select(c, upper_has, upper, lower);
        self.s.m.select(c, lower_has, lower, other)
    }

    /// The console wave's `op` over its lanes' terms `x` (integers), or with
    /// `prefix` each lane's prefix sum, from `r`, this host wave's (of `op`'s
    /// type). Two host waves exchange their results, where every lane runs
    /// ([`keeps_64`]); a single host wave's are the console wave's.
    pub(super) fn whole_wave(&mut self, op: WaveOp, prefix: bool, r: V, x: V) -> Result<V, Why> {
        if self.halves.as_ref().and_then(|h| h.two).is_none() {
            return Ok(r);
        }
        let (c, i32t) = (self.cur, self.s.t.i32);
        if prefix {
            match op {
                WaveOp::AddU => {}
                WaveOp::AddF => return Err(FLOAT_SUM),
                WaveOp::MinU
                | WaveOp::MaxU
                | WaveOp::MinI
                | WaveOp::MaxI
                | WaveOp::MinF
                | WaveOp::MaxF
                | WaveOp::Or
                | WaveOp::And
                | WaveOp::Xor => return Err("a wave prefix of an operation other than a sum"),
            }
            // The upper half's prefix goes on from the lower half's total.
            let (sum, unsigned) = (self.s.m.i8(0), self.s.m.i8(1));
            let total =
                self.s.op(c, "waveActiveOp.i32", Attr::NoUnwind, i32t, 119, &[x, sum, unsigned]);
            let (lower, _) = self.exchange(c, total);
            let lane = self.console_lane();
            let (k32, zero) = (self.s.m.i32(32), self.s.m.i32(0));
            let upper = self.s.m.cmp(c, pred::IUGE, lane, k32);
            let add = self.s.m.select(c, upper, lower, zero);
            return Ok(self.s.m.bin(c, Bin::Add, r, add));
        }
        let float = matches!(op, WaveOp::MinF | WaveOp::MaxF);
        let mine = if float { self.as_int(r) } else { r };
        let (lo, hi) = self.exchange(c, mine);
        // DXIL's IMax, IMin, UMax, UMin
        let code = match op {
            WaveOp::AddU => return Ok(self.s.m.bin(c, Bin::Add, lo, hi)),
            WaveOp::Or => return Ok(self.s.m.bin(c, Bin::Or, lo, hi)),
            WaveOp::And => return Ok(self.s.m.bin(c, Bin::And, lo, hi)),
            WaveOp::Xor => return Ok(self.s.m.bin(c, Bin::Xor, lo, hi)),
            WaveOp::MinF | WaveOp::MaxF => {
                let (lo, hi) = (self.as_float(lo), self.as_float(hi));
                return Ok(self.max_min(op == WaveOp::MaxF, lo, hi));
            }
            WaveOp::AddF => return Err(FLOAT_SUM),
            WaveOp::MaxI => 37,
            WaveOp::MinI => 38,
            WaveOp::MaxU => 39,
            WaveOp::MinU => 40,
        };
        Ok(self.s.op(c, "binary.i32", Attr::ReadNone, i32t, code, &[lo, hi]))
    }

    /// The condition of the branch closing IR block `b`, which the console
    /// wave takes as one ([`Structured::wide`]): taken when some lane's
    /// `when` holds, or only when every lane's does
    /// ([`Structured::wave_any`]). Each half asks its lanes and the halves
    /// exchange the answer, so both go the same way. `b` is a block every
    /// lane runs ([`to_widen`]).
    pub(super) fn wave_wide(&mut self, b: usize, when: Bool) -> Result<V, Why> {
        let any =
            self.x.st.wave_any.get(b).copied().flatten().ok_or("a wave-wide branch on a scalar")?;
        let c = self.cur;
        let mine = self.boolean(when, c)?;
        // What a half tells is whether some lane of it speaks against the
        // other way: for `any`, some lane's `when` holds; else some lane's
        // does not.
        let against = if any {
            mine
        } else {
            let t = self.s.m.bool(true);
            self.s.m.bin(c, Bin::Xor, mine, t)
        };
        let (i1t, i32t) = (self.s.t.i1, self.s.t.i32);
        let some = self.s.op(c, "waveAnyTrue", Attr::NoUnwind, i1t, 113, &[against]);
        self.s.flags |= flags::WAVE_OPS;
        let some = self.s.m.cast(c, Cast::ZExt, some, i32t);
        let (lo, hi) = self.exchange(c, some);
        let either = self.s.m.bin(c, Bin::Or, lo, hi);
        let zero = self.s.m.i32(0);
        Ok(self.s.m.cmp(c, if any { pred::NE } else { pred::EQ }, either, zero))
    }

    /// A group barrier with an LDS fence.
    pub(super) fn group_sync(&mut self, at: usize) {
        let mode = self.s.m.i32(1 | 8);
        let void = self.s.t.void;
        self.s.op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }
}

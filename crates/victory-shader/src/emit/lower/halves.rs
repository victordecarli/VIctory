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
//! - a group of two host waves has no LDS of its own (the console compiler
//!   leaves out the barriers one wave sharing LDS needs);
//! - it has no whole-wave section ([`super::super::wave`]) and no
//!   `ds_(b)permute`;
//! - in a group of two host waves, every place the halves exchange lies in
//!   a block every lane runs;
//! - a block only some lanes run holds no group barrier and writes no
//!   scalar other than a lane mask.
//!
//! Every other console wave64 program keeps 64-lane waves.
//!
//! Ported from the fxceless line (`base/fxceless`, `emit/lower.rs`:
//! `wave_base`, `console_thread`, `console_lane`, `exchange_pair`,
//! `exchange_halves`, `branch_regions`, `all_lane_blocks`), restricted to
//! the cases above.

use super::*;

/// LDS dwords a split program adds past its own: the counter numbering
/// the group's two host waves, then the exchange pair. None for a group of
/// a single host wave. Values made once, in the entry block.
pub(super) struct Halves {
    two: Option<(u32, u32)>,
    base: Option<V>,
    lane: Option<V>,
}

impl Halves {
    /// The halves of a group of `group` threads, taking their LDS dwords
    /// past `lds_dwords`.
    pub(super) fn new(lds_dwords: &mut u32, group: u32) -> Self {
        let two = (group > 32).then(|| {
            let counter = *lds_dwords;
            *lds_dwords += DWORDS;
            (counter, counter + 1)
        });
        Halves { two, base: None, lane: None }
    }

    /// The LDS dwords these halves add past the program's own.
    pub(super) fn dwords(&self) -> u32 {
        if self.two.is_some() { DWORDS } else { 0 }
    }
}

/// The counter and the exchange pair.
const DWORDS: u32 = 3;

/// Why the console wave64 compute program `p` (`group` threads, `lds_dwords`
/// of LDS) keeps 64-lane waves on a host of 32-lane waves; None when it runs
/// in halves.
pub(super) fn keeps_64(p: &Program, st: &Structured, group: u32, lds_dwords: u32) -> Option<Why> {
    let two = match group {
        0..=32 => false,
        64 => true,
        _ => return Some("a thread group of other than one console wave"),
    };
    if two && lds_dwords > 0 {
        return Some("LDS shared within a console wave of two host waves");
    }
    if (0..p.blocks.len()).any(|b| !super::super::wave::sections(p, b).is_empty()) {
        return Some("a whole-wave section");
    }
    let all = match all_lane_blocks(st) {
        Some(all) => all,
        None => return Some("an indirect exit"),
    };
    let every = |b: usize| all.get(b).copied().unwrap_or(false);
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
    for (b, block) in p.blocks.iter().enumerate() {
        for inst in &block.insts {
            let n = inst.name();
            if n.starts_with("ds_permute") || n.starts_with("ds_bpermute") {
                return Some("ds_(b)permute");
            }
            if two && !every(b) && matches!(n, "v_readlane_b32" | "v_readfirstlane_b32") {
                return Some("a lane read where not every lane runs");
            }
            if two && reads_mask_as_data(st, inst) {
                // The ballot is taken where the mask is written (`mask_dword`).
                let mut mask_blocks = mask_operands(inst).map(|v| {
                    let low = st.highs.get(&v).copied().unwrap_or(v);
                    // (program inputs, EXEC on entry, are the entry block's)
                    written.get(&low).copied().unwrap_or(0)
                });
                if mask_blocks.any(|m| !every(m)) {
                    return Some("a lane mask read as a number, written where not every lane runs");
                }
            }
            if every(b) {
                continue;
            }
            if n == "s_barrier" {
                return Some("a group barrier where not every lane runs");
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
                    return Some("a scalar written where not every lane runs");
                }
            }
        }
    }
    None
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
/// branch's region: the blocks reachable from the branch that do not
/// post-dominate it). None with an indirect exit, which is not followed.
pub(super) fn all_lane_blocks(st: &Structured) -> Option<Vec<bool>> {
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
    let mut all = vec![true; n];
    for (d, e) in st.exits.iter().enumerate() {
        let LaneExit::Branch { when, .. } = *e else { continue };
        if st.bools.support(when).iter().all(|a| matches!(a, Atom::Uniform(_))) {
            continue; // a scalar test: every lane goes the same way
        }
        // The blocks reachable from the branch that do not post-dominate it.
        let mut seen = vec![false; n];
        let mut stack: Vec<usize> = succ[d].clone();
        while let Some(b) = stack.pop() {
            if seen[b] || pdom[d][b] {
                continue;
            }
            seen[b] = true;
            all[b] = false;
            stack.extend(&succ[b]);
        }
    }
    Some(all)
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

    /// `v_readfirstlane`: where every lane runs (as [`keeps_64`] requires of
    /// a group of two host waves), the console wave's first active lane is
    /// lane 0, in the lower half.
    pub(super) fn console_first_lane(&mut self, x: V) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let mine = self.s.op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[x]);
        self.s.flags |= flags::WAVE_OPS;
        self.exchange(c, mine).0
    }

    /// A group barrier with an LDS fence.
    fn group_sync(&mut self, at: usize) {
        let mode = self.s.m.i32(1 | 8);
        let void = self.s.t.void;
        self.s.op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }
}

//! Thread groups of several console waves (128 to 1024 threads) in halves.
//!
//! Each console wave runs as two host waves, as in a group of one, but a
//! group barrier now waits for every console wave of the group, and those
//! decide their branches each for themselves. So a lane branch around a
//! place where the halves meet is taken by the whole group
//! ([`Lower::group_wide`]): the group enters its region when some console
//! wave would, and inside, each console wave runs its vector instructions
//! only if it would have entered: its decision is a flag added to every
//! guard there ([`Lower::group_flag`]), so a wave that would have skipped
//! the region keeps its VGPRs and stores nothing. Every console wave thus
//! meets the others at the same group barriers, the only ones that are
//! well defined. A scalar branch the console waves may take apart, around
//! such a place, is taken so too ([`apart_branches`]). A way that only ends
//! the program (`if (...) return;`) counts as the region's join
//! ([`group_region`]).
//!
//! That is exact only where the region has nothing else a skipping wave
//! could get wrong ([`several_keep_64`]): such a wave still runs the
//! region's scalar instructions, which no guard holds back.

use super::*;

/// The region of a lane branch the whole group takes: the way into it and,
/// per IR block, whether the block lies in it.
pub(super) struct GroupRegion {
    pub(super) enter: usize,
    pub(super) blocks: Vec<bool>,
}

/// The successors of each IR block, None with an indirect exit.
fn successors(st: &Structured) -> Option<Vec<Vec<usize>>> {
    st.exits
        .iter()
        .map(|e| match *e {
            LaneExit::Jump(t) => Some(vec![t]),
            LaneExit::Branch { taken, not_taken, .. } => Some(vec![taken, not_taken]),
            LaneExit::End => Some(Vec::new()),
            LaneExit::Indirect => None,
        })
        .collect()
}

/// The blocks reachable from `d`'s successors before the first block that
/// post-dominates `d`, and whether `d` is among them (a loop).
fn region_of(succ: &[Vec<usize>], pdom: &[Vec<bool>], d: usize) -> (Vec<bool>, bool) {
    let mut blocks = vec![false; succ.len()];
    let mut stack = succ[d].clone();
    let mut cycle = false;
    while let Some(b) = stack.pop() {
        // (`d` post-dominates itself: a way back to it is a loop all the same)
        cycle |= b == d;
        if blocks[b] || pdom[d][b] {
            continue;
        }
        blocks[b] = true;
        stack.extend(&succ[b]);
    }
    (blocks, cycle)
}

/// The region the whole group takes at the lane branch closing block `d`:
/// an `if` without an `else` (one way is the join), not closing a loop. A
/// way that only ends the program (`if (...) return;`, [`bare_end`]) joins
/// at the end: a console wave that would have ended there runs the rest
/// with the group, its flag guarding it, and ends with it.
pub(super) fn group_region(p: &Program, st: &Structured, d: usize) -> Result<GroupRegion, Why> {
    let succ = successors(st).ok_or("an indirect exit")?;
    let LaneExit::Branch { taken, not_taken, .. } = st.exits[d] else {
        return Err("a group-wide branch that is no branch");
    };
    let pdom = post_dominators(&succ);
    let (mut blocks, cycle) = region_of(&succ, &pdom, d);
    if cycle {
        return Err("a lane loop around a meeting in a group of several console waves");
    }
    let joins = |b: usize| pdom[d][b] || bare_end(p, st, b);
    let enter = match (joins(taken), joins(not_taken)) {
        (true, false) => not_taken,
        (false, true) => taken,
        _ => {
            return Err(
                "a lane branch with work on both ways around a meeting in a group of several console waves",
            );
        }
    };
    let end = if enter == taken { not_taken } else { taken };
    blocks[end] = false;
    Ok(GroupRegion { enter, blocks })
}

/// Whether IR block `b` only ends the program: `s_endpgm` alone.
fn bare_end(p: &Program, st: &Structured, b: usize) -> bool {
    st.exits[b] == LaneExit::End && p.blocks[b].insts.iter().all(|i| i.name() == "s_endpgm")
}

/// Whether `inst` is where the halves or the group meet: a lane read, a
/// lane mask read as a number, an LDS access (with LDS of the program's
/// own), or a group barrier. (Whole-wave sections are per block.)
fn meets(st: &Structured, inst: &Inst, lds: bool) -> bool {
    let n = inst.name();
    n == "s_barrier"
        || matches!(n, "v_readlane_b32" | "v_readfirstlane_b32")
        || reads_mask_as_data(st, inst)
        || lds && lds_access(inst)
}

/// Why `p`, whose group of `group` threads holds several console waves,
/// keeps 64-lane waves beyond what a group of one does ([`keeps_64`]).
pub(super) fn several_keep_64(
    p: &Program,
    st: &Structured,
    group: u32,
    lds_dwords: u32,
) -> Option<Keep> {
    // (the host's 32 KiB of LDS)
    if lds_dwords + dwords(group) > 8192 {
        return Keep::fixed("no room in LDS for the exchanges of several console waves");
    }
    let Some(apart) = apart_branches(p, st, lds_dwords > 0) else {
        return Keep::fixed("an indirect exit");
    };
    let mut taken: Vec<usize> = st.wide.iter().copied().chain(apart).collect();
    taken.sort_unstable();
    for &d in &taken {
        let region = match group_region(p, st, d) {
            Ok(r) => r,
            Err(why) => return Keep::fixed(why),
        };
        if let Some(keep) = skipping_wave_keeps_64(p, &region) {
            return Some(keep);
        }
    }
    None
}

/// The scalar branches of `p` the console waves of a group may take apart
/// (their test may differ between them, [`group_varying`]) around a place
/// where the halves or the group meet: the whole group takes them, as the
/// wide lane branches ([`Lower::group_wide`]). `lds`: the program has LDS
/// of its own. None with an indirect exit.
pub(super) fn apart_branches(p: &Program, st: &Structured, lds: bool) -> Option<Vec<usize>> {
    let succ = successors(st)?;
    let meet = |b: usize| {
        st.wide.contains(&b)
            || !super::super::super::wave::sections(p, b).is_empty()
            || p.blocks[b].insts.iter().any(|i| meets(st, i, lds))
    };
    let pdom = post_dominators(&succ);
    let varying = group_varying(p, st, &succ, &pdom);
    let mut apart = Vec::new();
    for (d, e) in st.exits.iter().enumerate() {
        let LaneExit::Branch { when, .. } = *e else { continue };
        let support = st.bools.support(when);
        if st.wide.contains(&d) || !support.iter().all(|a| matches!(a, Atom::Uniform(_))) {
            continue; // a lane branch: the rules of a group of one hold
        }
        let varies = support.iter().any(|a| matches!(a, Atom::Uniform(v) if varying.contains(v)));
        let (blocks, _) = region_of(&succ, &pdom, d);
        if varies && (0..blocks.len()).any(|b| blocks[b] && meet(b)) {
            apart.push(d);
        }
    }
    Some(apart)
}

/// What a console wave that would have skipped `region`, but runs its
/// scalar instructions, could get wrong: a scalar store or atomic, a lane
/// write (which ignores EXEC), a group barrier of the program's own, or a
/// scalar written there and read outside it (EXEC aside: inside, a
/// skipping wave's lanes stay out of it, and it is restored as it was).
fn skipping_wave_keeps_64(p: &Program, region: &GroupRegion) -> Option<Keep> {
    let inside = |b: usize| region.blocks.get(b).copied().unwrap_or(false);
    let mut written = HashSet::new();
    for (b, block) in p.blocks.iter().enumerate().filter(|&(b, _)| inside(b)) {
        // (a whole-wave section's results are written whatever the guard)
        for section in super::super::super::wave::sections(p, b).values() {
            written.extend(section.shapes.keys().copied());
        }
        for inst in &block.insts {
            let n = inst.name();
            let scalar_store = matches!(inst.fields, Fields::Smem { .. })
                && (n.contains("store") || n.contains("atomic"));
            if scalar_store || n == "v_writelane_b32" || n == "s_barrier" {
                return Keep::fixed(
                    "a group-wide branch around a scalar store, a lane write or a group barrier",
                );
            }
            for &(r, w) in &inst.writes {
                if matches!(r, Reg::S(c) if c != EXEC && c != EXEC + 1) || r == Reg::Scc {
                    written.insert(w);
                }
            }
        }
        for phi in &block.phis {
            if !matches!(phi.reg, Reg::S(c) if c == EXEC || c == EXEC + 1)
                && !matches!(phi.reg, Reg::V(_))
            {
                written.insert(phi.value);
            }
        }
    }
    let outside_reads =
        p.blocks.iter().enumerate().filter(|&(b, _)| !inside(b)).flat_map(|(_, block)| {
            let insts = block.insts.iter().flat_map(|i| i.reads.iter().map(|r| r.1));
            insts.chain(block.phis.iter().flat_map(|phi| phi.args.iter().copied()))
        });
    for v in outside_reads {
        if written.contains(&v) {
            return Keep::fixed("a scalar written in a group-wide branch, read past it");
        }
    }
    None
}

/// The values that may differ between the console waves of a group: what
/// comes from a lane (VGPRs, lane masks and what reads them), LDS, atomics,
/// the hardware's ids and clocks, and joins of scalar branches that depend
/// on those. Scalar inputs (user data, the group id) are the group's.
fn group_varying(
    p: &Program,
    st: &Structured,
    succ: &[Vec<usize>],
    pdom: &[Vec<bool>],
) -> HashSet<Value> {
    let mut varying: HashSet<Value> =
        p.inputs.iter().filter(|(r, _)| matches!(r, Reg::V(_))).map(|&(_, v)| v).collect();
    // Scalar branches and the blocks they join in: their region and the
    // first block past it on each way out.
    let scalar_branches: Vec<(Vec<Value>, Vec<bool>)> = st
        .exits
        .iter()
        .enumerate()
        .filter_map(|(d, e)| {
            let LaneExit::Branch { when, .. } = *e else { return None };
            let support = st.bools.support(when);
            let values: Vec<Value> = support
                .iter()
                .map(|a| match *a {
                    Atom::Uniform(v) => Some(v),
                    _ => None,
                })
                .collect::<Option<_>>()?;
            let (mut blocks, _) = region_of(succ, pdom, d);
            let past: Vec<usize> = (0..blocks.len())
                .filter(|&b| blocks[b] || b == d)
                .flat_map(|b| succ[b].iter().copied())
                .filter(|&t| !blocks[t])
                .collect();
            for t in past {
                blocks[t] = true;
            }
            Some((values, blocks))
        })
        .collect();
    loop {
        let before = varying.len();
        let apart: Vec<&Vec<bool>> = scalar_branches
            .iter()
            .filter(|(values, _)| values.iter().any(|v| varying.contains(v)))
            .map(|(_, blocks)| blocks)
            .collect();
        for (b, block) in p.blocks.iter().enumerate() {
            let joined_apart = apart.iter().any(|blocks| blocks[b]);
            for phi in &block.phis {
                if joined_apart || phi.args.iter().any(|a| varying.contains(a)) {
                    varying.insert(phi.value);
                }
            }
            for inst in &block.insts {
                let n = inst.name();
                let source = n.starts_with("v_")
                    || matches!(inst.fields, Fields::Ds { .. })
                    || n.contains("atomic")
                    || n.starts_with("s_getreg")
                    || n.starts_with("s_memtime")
                    || n.starts_with("s_memrealtime");
                if source || inst.reads.iter().any(|r| varying.contains(&r.1)) {
                    varying.extend(inst.writes.iter().map(|w| w.1));
                }
            }
        }
        if varying.len() == before {
            return varying;
        }
    }
}

impl Lower<'_> {
    /// The branch closing block `b` taken by the whole group
    /// ([`group_region`]), a wide lane branch or a scalar one the console
    /// waves may take apart ([`apart_branches`]): each host wave gives `some`
    /// (0 or 1, whether some lane of it speaks against the other way, as
    /// [`Self::wave_wide`] has it for `any`; for a scalar branch, its test),
    /// so each console wave's way is its halves' answer; the
    /// group enters the region when some console wave would. This console
    /// wave's own answer becomes the flag of the region's blocks. The
    /// condition for the way taken.
    pub(in super::super) fn group_wide(&mut self, b: usize, any: bool, some: V) -> Result<V, Why> {
        let region = group_region(self.x.p, self.x.st, b)?;
        let LaneExit::Branch { taken, .. } = self.x.st.exits[b] else {
            return Err("a group-wide branch that is no branch");
        };
        let (Some((_, pair)), waves) = self.halves.as_ref().map_or((None, 0), |h| (h.two, h.waves))
        else {
            return Err("a group-wide branch outside halves");
        };
        let c = self.cur;
        self.post(c, pair, some);
        let lds = self.s.lds();
        let flip = region.enter != taken;
        let mut group = self.s.m.bool(false);
        for w in 0..waves {
            let (k0, k1) = (self.s.m.i32(pair + 2 * w), self.s.m.i32(pair + 2 * w + 1));
            let (p0, p1) = (self.s.m.element(c, lds, k0), self.s.m.element(c, lds, k1));
            let (lo, hi) = (self.s.m.load(c, p0), self.s.m.load(c, p1));
            let enters = self.wave_enters(lo, hi, any, flip);
            group = self.s.m.bin(c, Bin::Or, group, enters);
        }
        // This console wave's pair: slots 2w and 2w + 1 past the first.
        let thread = self.console_thread();
        let (six, one) = (self.s.m.i32(6), self.s.m.i32(1));
        let wave = self.s.m.bin(c, Bin::LShr, thread, six);
        let even = self.s.m.bin(c, Bin::Shl, wave, one);
        let first = self.s.m.i32(pair);
        let lower = self.s.m.bin(c, Bin::Add, first, even);
        let upper = self.s.m.bin(c, Bin::Add, lower, one);
        let (p0, p1) = (self.s.m.element(c, lds, lower), self.s.m.element(c, lds, upper));
        let (lo, hi) = (self.s.m.load(c, p0), self.s.m.load(c, p1));
        let mut own = self.wave_enters(lo, hi, any, flip);
        // The slots are written again only after every thread has read them.
        self.group_sync(c);
        if let Some(outer) = self.halves.as_ref().and_then(|h| h.flags.get(&b).copied()) {
            own = self.s.m.bin(c, Bin::And, own, outer);
        }
        if let Some(h) = self.halves.as_mut() {
            for (blk, &inside) in region.blocks.iter().enumerate() {
                if inside {
                    h.flags.insert(blk, own);
                }
            }
        }
        Ok(if flip { self.not(group) } else { group })
    }

    /// Whether a console wave whose halves answered `lo` and `hi` enters
    /// the region: takes the branch (`flip`: does not).
    fn wave_enters(&mut self, lo: V, hi: V, any: bool, flip: bool) -> V {
        let c = self.cur;
        let either = self.s.m.bin(c, Bin::Or, lo, hi);
        let zero = self.s.m.i32(0);
        let take = self.s.m.cmp(c, if any { pred::NE } else { pred::EQ }, either, zero);
        if flip { self.not(take) } else { take }
    }

    fn not(&mut self, x: V) -> V {
        let t = self.s.m.bool(true);
        self.s.m.bin(self.cur, Bin::Xor, x, t)
    }

    /// `guard` of an instruction in IR block `b`, with the flag of the
    /// region the whole group takes around it ([`Self::group_wide`]).
    pub(in super::super) fn group_flag(&mut self, b: usize, guard: Option<V>) -> Option<V> {
        let Some(f) = self.halves.as_ref().and_then(|h| h.flags.get(&b).copied()) else {
            return guard;
        };
        Some(match guard {
            Some(g) => self.s.m.bin(self.cur, Bin::And, g, f),
            None => f,
        })
    }
}

impl Lower<'_> {
    /// Whether the scalar branch closing block `b` is one the whole group
    /// takes ([`apart_branches`]).
    pub(in super::super) fn group_apart(&mut self, b: usize) -> bool {
        let Some(h) = self.halves.as_mut().filter(|h| h.waves > 1) else { return false };
        if h.apart.is_none() {
            let lds = h.lds;
            h.apart = Some(apart_branches(self.x.p, self.x.st, lds).unwrap_or_default());
        }
        self.halves.as_ref().and_then(|h| h.apart.as_ref()).is_some_and(|a| a.contains(&b))
    }
}

impl Lower<'_> {
    /// The scalar branch closing block `b`, on `when`, taken by the whole
    /// group: a console wave's halves have the same scalars, so each host
    /// wave gives the test as its own answer.
    pub(in super::super) fn group_scalar(&mut self, b: usize, when: Bool) -> Result<V, Why> {
        let c = self.cur;
        let test = self.boolean(when, c)?;
        let i32t = self.s.t.i32;
        let some = self.s.m.cast(c, Cast::ZExt, test, i32t);
        self.group_wide(b, true, some)
    }
}

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
//! the region keeps its VGPRs and stores nothing, and it accesses no memory
//! there ([`Lower::flagged`]): its scalars may hold anything, an index past
//! the descriptor heap among them. Every console wave thus meets the others
//! at the same group barriers, the only ones that are well defined, the
//! program's own included. A scalar branch the console waves may take apart, around
//! such a place, is taken so too ([`apart_branches`]). A way that only ends
//! the program (`if (...) return;`) counts as the region's join
//! ([`group_region`]). An `if` with an `else` is taken where only one way
//! has such a place ([`group_if_else`]): the group runs that way as the
//! region, then each console wave that takes the other, quiet, way runs it
//! on its own, by a branch of its own ([`Lower::group_else`]), so each
//! comes to the join by its own way.
//!
//! A loop around such a place whose branch is its one way out, passed every
//! round, is taken so too ([`group_loop`]): the group goes round while some
//! console wave would, and a wave's flag in the loop, a phi at its header,
//! turns off once the wave would have left ([`Lower::group_round`]), so it
//! keeps its VGPRs from then on. A wave already out votes to leave. A loop
//! inside such a loop is taken so too: a wave enters it with its flag in
//! the loop around.
//!
//! That is exact only where the region has nothing else a skipping wave
//! could get wrong ([`several_keep_64`]): such a wave still runs the
//! region's scalar instructions, which no guard holds back. A scalar the
//! `if`'s join takes from inside it is each wave's own there: a wave that
//! would have skipped the `if` takes what the join takes from its branch
//! ([`Lower::group_join`]).

use super::*;

/// The region of a lane branch the whole group takes: the way into it and,
/// per IR block, whether the block lies in it.
pub(super) struct GroupRegion {
    pub(super) enter: usize,
    pub(super) blocks: Vec<bool>,
    /// For an `if`: the block its branch closes and the block its ways join
    /// at, whose scalar phis each console wave takes by its own way
    /// ([`Lower::group_join`]). None for a loop.
    pub(super) join: Option<(usize, usize)>,
    /// For an `if` with an `else` where the halves and the group meet on one
    /// way only (the region): where the other, quiet, way begins, which each
    /// console wave that takes it runs on its own once the group ran the
    /// region ([`Lower::group_target`]).
    pub(super) quiet: Option<usize>,
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
/// post-dominates `d`, and whether `d` is among them (a loop). In a loop,
/// `d` itself runs once per round, so it is one of them.
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
    blocks[d] |= cycle;
    (blocks, cycle)
}

/// The region the whole group takes at the lane branch closing block `d`:
/// an `if` without an `else` (one way is the join), not closing a loop. A
/// way that only ends the program (`if (...) return;`, [`bare_end`]) joins
/// at the end: a console wave that would have ended there runs the rest
/// with the group, its flag guarding it, and ends with it. An `if` with an
/// `else` is taken where only one way meets ([`group_if_else`]). `lds`: the
/// program has LDS of its own.
pub(super) fn group_region(
    p: &Program,
    st: &Structured,
    d: usize,
    lds: bool,
) -> Result<GroupRegion, Why> {
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
        (false, false) => return group_if_else(p, st, &succ, &pdom, d, lds),
        (true, true) => return Err(BOTH_WAYS),
    };
    let end = if enter == taken { not_taken } else { taken };
    blocks[end] = false;
    Ok(GroupRegion { enter, blocks, join: Some((d, end)), quiet: None })
}

const BOTH_WAYS: Why =
    "a lane branch with work on both ways around a meeting in a group of several console waves";

/// The region the whole group takes at the branch closing block `d`, an
/// `if` with an `else` whose ways join at the block first post-dominating
/// `d`: the way where the halves or the group meet, the other way quiet
/// (no such place, so no branch the group takes either). Each way begins
/// at a block only `d` leads to, and the join is reached from the ways
/// alone.
fn group_if_else(
    p: &Program,
    st: &Structured,
    succ: &[Vec<usize>],
    pdom: &[Vec<bool>],
    d: usize,
    lds: bool,
) -> Result<GroupRegion, Why> {
    let LaneExit::Branch { taken, not_taken, .. } = st.exits[d] else {
        return Err("a group-wide branch that is no branch");
    };
    let n = succ.len();
    let first = |c: usize| (0..n).all(|o| o == d || !pdom[d][o] || pdom[c][o]);
    let join = (0..n).find(|&c| c != d && pdom[d][c] && first(c)).ok_or(BOTH_WAYS)?;
    let way = |s: usize| {
        let mut blocks = vec![false; n];
        let mut stack = vec![s];
        while let Some(b) = stack.pop() {
            if blocks[b] || pdom[d][b] {
                continue;
            }
            blocks[b] = true;
            stack.extend(&succ[b]);
        }
        blocks
    };
    let (t, f) = (way(taken), way(not_taken));
    let meet = |w: &[bool]| (0..n).any(|b| w[b] && meets_in(p, st, b, lds));
    let (enter, quiet, blocks) = match (meet(&t), meet(&f)) {
        (true, false) => (taken, not_taken, t.clone()),
        (false, true) => (not_taken, taken, f.clone()),
        (true, true) | (false, false) => return Err(BOTH_WAYS),
    };
    let only_from_d = |b: usize| p.blocks[b].preds == [d];
    let apart = (0..n).all(|b| !(t[b] && f[b]));
    let joined = p.blocks[join].preds.iter().all(|&q| t[q] || f[q]);
    if !(apart && joined && only_from_d(enter) && only_from_d(quiet)) {
        return Err(BOTH_WAYS);
    }
    Ok(GroupRegion { enter, blocks, join: Some((d, join)), quiet: Some(quiet) })
}

/// Whether IR block `b` is where the halves or the group meet: a branch
/// the console wave takes as one, a whole-wave section, or an instruction
/// that [`meets`].
fn meets_in(p: &Program, st: &Structured, b: usize, lds: bool) -> bool {
    st.wide.contains(&b)
        || !super::super::super::wave::sections(p, b).is_empty()
        || p.blocks[b].insts.iter().any(|i| meets(st, i, lds))
}

/// Whether IR block `b` only ends the program: `s_endpgm` alone.
fn bare_end(p: &Program, st: &Structured, b: usize) -> bool {
    st.exits[b] == LaneExit::End && p.blocks[b].insts.iter().all(|i| i.name() == "s_endpgm")
}

/// A loop the whole group takes at a branch closing one of its blocks, its
/// only way out ([`group_loop`]).
pub(super) struct GroupLoop {
    /// The block every round starts at.
    header: usize,
    /// Per IR block: whether it lies in the loop.
    blocks: Vec<bool>,
    /// Per IR block: in the loop past the branch, before the round ends.
    after: Vec<bool>,
}

/// The loop around the branch closing block `d`, None when `d` is in none:
/// the innermost loop holding `d` that `d` leaves ([`loop_left_at`]). The
/// whole group takes it only where the branch is its one way out and every
/// round passes the branch, so that each round a console wave goes on or
/// leaves there. A loop inside it is taken so too, each with its own flag.
pub(super) fn group_loop(p: &Program, st: &Structured, d: usize) -> Result<Option<GroupLoop>, Why> {
    let succ = successors(st).ok_or("an indirect exit")?;
    let LaneExit::Branch { taken, not_taken, .. } = st.exits[d] else {
        return Err("a group-wide branch that is no branch");
    };
    let n = succ.len();
    let mut preds = vec![Vec::new(); n];
    for (b, to) in succ.iter().enumerate() {
        for &t in to {
            preds[t].push(b);
        }
    }
    // A branch inside a loop goes back to itself too, round the loop: only
    // one whose region (up to the block joining its ways) does is a loop's.
    let (_, cycle) = region_of(&succ, &post_dominators(&succ), d);
    if !cycle {
        return Ok(None);
    }
    let blocks = loop_left_at(p, &succ, &preds, d)?;
    let leaves = |b: usize| succ[b].iter().any(|&t| !blocks[t]);
    if (0..n).any(|b| blocks[b] && b != d && leaves(b)) {
        return Err("a group-wide loop left at more than one place");
    }
    let inside = match (blocks[taken], blocks[not_taken]) {
        (true, false) => taken,
        (false, true) => not_taken,
        _ => return Err("a group-wide loop branch with no way out"),
    };
    let entered: Vec<usize> =
        (0..n).filter(|&b| blocks[b] && preds[b].iter().any(|&q| !blocks[q])).collect();
    let &[header] = &entered[..] else {
        return Err("a group-wide loop entered at more than one block");
    };
    let mut after = vec![false; n];
    let mut stack = vec![inside];
    while let Some(b) = stack.pop() {
        if b == header || after[b] {
            continue;
        }
        after[b] = true;
        stack.extend(succ[b].iter().filter(|&&t| blocks[t]));
    }
    if (0..n).any(|b| blocks[b] && succ[b].contains(&header) && b != d && !after[b]) {
        return Err("a group-wide loop that goes round without its branch");
    }
    Ok(Some(GroupLoop { header, blocks, after }))
}

/// Per IR block of `p`: whether it lies in the innermost loop holding block
/// `d` that `d` leaves. A loop is a natural one: its header dominates the
/// blocks going back to it, and it holds the blocks reaching those without
/// passing the header (every way back to the header together). The blocks
/// reaching `d` and reached from it would take in, for a loop inside
/// another, the outer one too.
fn loop_left_at(
    p: &Program,
    succ: &[Vec<usize>],
    preds: &[Vec<usize>],
    d: usize,
) -> Result<Vec<bool>, Why> {
    let idom = idoms(p);
    let n = succ.len();
    let mut innermost: Option<(usize, Vec<bool>)> = None;
    let mut held = false;
    for h in 0..n {
        let back: Vec<usize> =
            preds[h].iter().copied().filter(|&t| dominates(&idom, h, t)).collect();
        if back.is_empty() {
            continue;
        }
        let mut body = vec![false; n];
        body[h] = true;
        let mut stack = back;
        while let Some(b) = stack.pop() {
            if !std::mem::replace(&mut body[b], true) {
                // (blocks the entry does not reach are in no loop)
                stack.extend(preds[b].iter().filter(|&&q| idom[q] != usize::MAX));
            }
        }
        if !body[d] {
            continue;
        }
        held = true;
        let size = body.iter().filter(|&&x| x).count();
        if succ[d].iter().any(|&t| !body[t]) && innermost.as_ref().is_none_or(|(s, _)| size < *s) {
            innermost = Some((size, body));
        }
    }
    match innermost {
        Some((_, body)) => Ok(body),
        None if held => Err("a group-wide loop branch with no way out"),
        None => Err("a group-wide loop entered at more than one block"),
    }
}

/// Whether `inst` accesses memory other than LDS: what [`Lower::flagged`]
/// runs only where the console wave would.
fn memory_access(inst: &Inst) -> bool {
    match inst.fields {
        Fields::Smem { .. } | Fields::Image { .. } | Fields::Buffer { .. } => true,
        Fields::Flat { .. } => inst.name().starts_with("global_"),
        Fields::Sop { .. }
        | Fields::Sopp { .. }
        | Fields::Valu(_)
        | Fields::Vintrp { .. }
        | Fields::Ds { .. }
        | Fields::Exp { .. } => false,
    }
}

/// Whether `inst` reads a lane mask other than EXEC (which every vector
/// instruction reads) as an operand.
fn reads_mask(st: &Structured, inst: &Inst) -> bool {
    inst.reads.iter().any(|&(r, v)| {
        !matches!(r, Reg::S(c) if c == EXEC || c == EXEC + 1)
            && (st.masks.contains_key(&v) || st.highs.contains_key(&v))
    })
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
    let mut loops = Vec::new();
    for &d in &taken {
        match group_loop(p, st, d) {
            Ok(Some(lp)) => loops.push((d, lp)),
            Ok(None) => {}
            Err(why) => return Keep::fixed(why),
        }
    }
    for &d in &taken {
        // (a console wave out of a loop runs its later rounds as one that skips them)
        let region = match loops.iter().find(|(e, _)| *e == d) {
            Some((_, lp)) => {
                GroupRegion { enter: lp.header, blocks: lp.blocks.clone(), join: None, quiet: None }
            }
            None => match group_region(p, st, d, lds_dwords > 0) {
                Ok(r) => r,
                Err(why) => return Keep::fixed(why),
            },
        };
        if let Some(keep) = skipping_wave_keeps_64(p, st, &region) {
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
    let meet = |b: usize| meets_in(p, st, b, lds);
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
/// write (which ignores EXEC), or a scalar written there and read outside
/// it (EXEC aside: inside, a skipping wave's lanes stay out of it, and it
/// is restored as it was), but by a scalar phi (no lane mask) where an
/// `if`'s ways join, which takes each wave's own way
/// ([`Lower::group_join`]; its branch must dominate the way in, whose
/// values the phi's take). A group barrier of the program's own is no
/// trouble: every console wave reaches it, with the group. Nor is a memory
/// access ([`Lower::flagged`]), but for one reading a lane mask as a
/// number, whose ballot could have the halves meet where only some console
/// waves run.
fn skipping_wave_keeps_64(p: &Program, st: &Structured, region: &GroupRegion) -> Option<Keep> {
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
            if scalar_store || n == "v_writelane_b32" {
                return Keep::fixed("a group-wide branch around a scalar store or a lane write");
            }
            if memory_access(inst) && reads_mask(st, inst) {
                return Keep::fixed("a memory access reading a lane mask in a group-wide branch");
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
    // A scalar phi joining an `if`'s ways takes each console wave's own way
    // ([`Lower::group_join`]).
    let idom = region.join.map(|_| idoms(p));
    let joined = |b: usize, phi: &crate::ir::Phi, n: usize| {
        region.join.is_some_and(|(d, end)| {
            let q = p.blocks[b].preds[n];
            let scalar = (matches!(phi.reg, Reg::S(c) if c != EXEC && c != EXEC + 1)
                || phi.reg == Reg::Scc)
                && !st.masks.contains_key(&phi.value)
                && !st.highs.contains_key(&phi.value);
            b == end
                && inside(q)
                && phi.args.len() == p.blocks[b].preds.len()
                // (past an `else`, each wave comes by its own way)
                && (region.quiet.is_some()
                    || scalar
                        && p.blocks[b].preds.contains(&d)
                        && idom.as_ref().is_some_and(|idom| dominates(idom, d, q)))
        })
    };
    for (b, block) in p.blocks.iter().enumerate().filter(|&(b, _)| !inside(b)) {
        let insts = block.insts.iter().flat_map(|i| i.reads.iter().map(|r| r.1));
        let phis = block.phis.iter().flat_map(|phi| {
            phi.args.iter().enumerate().filter(|&(n, _)| !joined(b, phi, n)).map(|(_, &a)| a)
        });
        if insts.chain(phis).any(|v| written.contains(&v)) {
            return Keep::fixed("a scalar written in a group-wide branch, read past it");
        }
    }
    None
}

/// Each IR block's immediate dominator in `p` ([`crate::ir::idoms`]).
fn idoms(p: &Program) -> Vec<usize> {
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut index = vec![usize::MAX; p.blocks.len()];
    for (k, &b) in order.iter().enumerate() {
        index[b] = k;
    }
    crate::ir::idoms(&p.blocks, &order, &index)
}

/// Whether block `h` dominates block `b`, by their immediate dominators.
fn dominates(idom: &[usize], h: usize, mut b: usize) -> bool {
    loop {
        if b == h {
            return true;
        }
        if idom[b] == b || idom[b] == usize::MAX {
            return false;
        }
        b = idom[b];
    }
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
    /// wave's own answer becomes the flag of the region's blocks. A branch
    /// closing a loop the group takes is its way out ([`Self::group_round`]).
    /// The condition for the way taken.
    pub(in super::super) fn group_wide(&mut self, b: usize, any: bool, some: V) -> Result<V, Why> {
        if self
            .halves
            .as_ref()
            .is_some_and(|h| h.loops.as_ref().is_some_and(|l| l.contains_key(&b)))
        {
            return self.group_round(b, any, some);
        }
        let lds = self.halves.as_ref().is_some_and(|h| h.lds);
        let region = group_region(self.x.p, self.x.st, b, lds)?;
        let LaneExit::Branch { taken, .. } = self.x.st.exits[b] else {
            return Err("a group-wide branch that is no branch");
        };
        let flip = region.enter != taken;
        let (group, mut own) = self.group_vote(any, flip, some)?;
        if let Some(outer) = self.halves.as_ref().and_then(|h| h.flags.get(&b).copied()) {
            own = self.s.m.bin(self.cur, Bin::And, own, outer);
        }
        if let Some(h) = self.halves.as_mut() {
            for (blk, &inside) in region.blocks.iter().enumerate() {
                if inside {
                    h.flags.insert(blk, own);
                }
            }
        }
        match (region.join, region.quiet) {
            (Some((d, end)), Some(quiet)) => self.group_else(d, end, quiet, region.blocks, own)?,
            (Some((d, end)), None) => {
                if let Some(h) = self.halves.as_mut() {
                    h.joins.entry(end).or_default().push(Join { d, blocks: region.blocks, own });
                }
            }
            (None, _) => {}
        }
        Ok(if flip { self.not(group) } else { group })
    }

    /// The `else` of the `if` at block `d` ([`group_if_else`]): past the
    /// region (`arm`), whose ways to `join` go to a block of their own, each
    /// console wave goes on by its own decision `own`: to `join`, or into
    /// the quiet way, which it runs on its own. The group comes there
    /// straight from `d` when no console wave enters the region.
    fn group_else(
        &mut self,
        d: usize,
        join: usize,
        quiet: usize,
        arm: Vec<bool>,
        own: V,
    ) -> Result<(), Why> {
        let (Some(j), Some(q)) = (self.block[join], self.block[quiet]) else {
            return Err("an if with an else joining where the entry does not reach");
        };
        let (skip, dispatch) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(skip, Term::Br(dispatch));
        self.s.m.terminate(dispatch, Term::CondBr(own, j, q));
        let Some(h) = self.halves.as_mut() else {
            return Err("a group-wide branch outside halves");
        };
        if h.arms.contains_key(&join) {
            return Err("two ifs with an else the group takes joining at one block");
        }
        h.arms.insert(join, Arms { d, quiet, arm, skip, dispatch });
        Ok(())
    }

    /// The DXIL block a way from IR block `from` to `to` goes to instead of
    /// `to`'s, around an `if` with an `else` the whole group takes
    /// ([`Self::group_else`]): from its branch into the quiet way, the block
    /// the group comes by when no console wave enters the region; from the
    /// region to the join, the block where each console wave goes its own
    /// way.
    pub(in super::super) fn group_target(&self, from: usize, to: usize) -> Option<usize> {
        let arms = &self.halves.as_ref()?.arms;
        if let Some(a) = arms.get(&to).filter(|a| a.arm.get(from).copied().unwrap_or(false)) {
            return Some(a.dispatch);
        }
        arms.values().find(|a| a.d == from && a.quiet == to).map(|a| a.skip)
    }

    /// The incoming values of a phi of IR block `j`, each with its DXIL
    /// block and its IR block. Where `j` joins an `if` with an `else` the
    /// whole group takes, those from the region come to the join through
    /// the block where each console wave goes its own way, by a phi there
    /// (`zero` over the way the group takes when no console wave enters the
    /// region: no console wave goes on from there to the join).
    pub(in super::super) fn group_arms(
        &mut self,
        j: usize,
        incoming: Vec<(V, usize, usize)>,
        zero: V,
    ) -> Vec<(V, usize)> {
        let Some(a) = self.halves.as_ref().and_then(|h| h.arms.get(&j)) else {
            return incoming.into_iter().map(|(x, at, _)| (x, at)).collect();
        };
        let (skip, dispatch) = (a.skip, a.dispatch);
        let (inside, mut outside): (Vec<_>, Vec<_>) =
            incoming.into_iter().partition(|&(_, _, q)| a.arm.get(q).copied().unwrap_or(false));
        let ty = self.s.m.type_of(zero);
        let phi = self.s.m.phi(dispatch, ty);
        let mut through = vec![(zero, skip)];
        through.extend(inside.into_iter().map(|(x, at, _)| (x, at)));
        self.s.m.set_phi(phi, through);
        outside.push((phi, dispatch, j));
        outside.into_iter().map(|(x, at, _)| (x, at)).collect()
    }

    /// The group's vote at a branch it takes as one: each host wave posts
    /// `some`, and a console wave enters (takes the branch; `flip`: does
    /// not) by its halves' answers ([`Self::wave_enters`]). Whether some
    /// console wave enters, and whether this one does.
    fn group_vote(&mut self, any: bool, flip: bool, some: V) -> Result<(V, V), Why> {
        let (Some((_, pair)), waves) = self.halves.as_ref().map_or((None, 0), |h| (h.two, h.waves))
        else {
            return Err("a group-wide branch outside halves");
        };
        #[cfg(test)]
        {
            self.group_votes += 1;
        }
        let c = self.cur;
        self.post(c, pair, some);
        let lds = self.s.lds();
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
        let own = self.wave_enters(lo, hi, any, flip);
        // The slots are written again only after every thread has read them.
        self.group_sync(c);
        Ok((group, own))
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
    /// `lower`, the lowering of `inst`, an access to memory in IR block `b`
    /// ([`memory_access`]), run only by a console wave that would have
    /// entered the group-wide region around it ([`Self::group_wide`]): one
    /// that would have skipped it may hold anything in its scalars there, so
    /// it makes no handle and reads nothing. What `inst` writes keeps, where
    /// it did not run, a VGPR's old value, and zero for anything else (a
    /// scalar the wave never reads, [`skipping_wave_keeps_64`]).
    pub(in super::super) fn flagged(
        &mut self,
        b: usize,
        inst: &Inst,
        lower: impl FnOnce(&mut Self) -> Result<(), Why>,
    ) -> Result<(), Why> {
        let Some(flag) = self.halves.as_ref().and_then(|h| h.flags.get(&b).copied()) else {
            return lower(self);
        };
        // Made before the branch, so that they hold on the way around it.
        let mut outside = Vec::with_capacity(inst.writes.len());
        for &(r, w) in &inst.writes {
            let old = match (r, inst.read(r).map(|o| self.i32_of(o))) {
                (Reg::V(_), Some(Ok(o))) => o,
                // The validator refuses reads of undef.
                _ => self.s.m.i32(0),
            };
            outside.push((w, old));
        }
        let at = self.begin_guarded(Some(flag));
        #[cfg(test)]
        {
            self.flagged_accesses += u32::from(at.is_some());
        }
        let lowered = lower(self);
        let mut joins = Vec::new();
        let mut written = Vec::new();
        for (w, old) in outside {
            let (x, old, bit) = match self.vals.get(&w).copied() {
                Some(Val::I32(x)) => (x, old, false),
                Some(Val::Bit(x)) => (x, self.s.m.bool(false), true),
                Some(Val::Missing(_)) | None => continue,
            };
            joins.push((x, old));
            written.push((w, bit));
        }
        let joined = self.end_guarded(at, &joins);
        for ((w, bit), v) in written.into_iter().zip(joined) {
            self.vals.insert(w, if bit { Val::Bit(v) } else { Val::I32(v) });
        }
        lowered
    }
}

/// An `if` with an `else` the whole group takes ([`Lower::group_else`]), at
/// the block its ways join at.
pub(super) struct Arms {
    /// The block its branch closes.
    d: usize,
    /// The block its quiet way begins at.
    quiet: usize,
    /// Per IR block: whether it lies in the region, the way the group runs.
    arm: Vec<bool>,
    /// The DXIL block the group comes by from `d` when no console wave
    /// enters the region.
    skip: usize,
    /// The DXIL block where each console wave goes its own way.
    dispatch: usize,
}

/// An `if` the whole group takes ([`Lower::group_wide`]), at the block its
/// ways join at.
pub(super) struct Join {
    /// The block its branch closes.
    d: usize,
    /// Per IR block: whether it lies in the `if`.
    blocks: Vec<bool>,
    /// Whether this console wave enters it.
    own: V,
}

impl Lower<'_> {
    /// `x`, what the scalar phi `i` of IR block `j` takes over the way from
    /// block `q` (in DXIL block `at`), as each console wave takes it: where
    /// `j` joins the ways of an `if` the whole group takes and `q` lies in
    /// it, a console wave that would have skipped the `if` takes, instead,
    /// what the phi takes from the `if`'s branch. It ran the `if`'s scalar
    /// instructions with the group ([`skipping_wave_keeps_64`]). Innermost
    /// `if` first.
    pub(in super::super) fn group_join(
        &mut self,
        j: usize,
        i: usize,
        q: usize,
        at: usize,
        x: V,
    ) -> Result<V, Why> {
        let phi = &self.x.p.blocks[j].phis[i];
        let scalar = matches!(phi.reg, Reg::S(c) if c != EXEC && c != EXEC + 1);
        if !scalar && phi.reg != Reg::Scc {
            return Ok(x);
        }
        let Some(joins) = self.halves.as_ref().and_then(|h| h.joins.get(&j)) else {
            return Ok(x);
        };
        let mut around: Vec<(usize, usize, V)> = joins
            .iter()
            .filter(|r| r.blocks.get(q).copied().unwrap_or(false))
            .map(|r| (r.blocks.iter().filter(|&&b| b).count(), r.d, r.own))
            .collect();
        around.sort_unstable_by_key(|&(size, d, _)| (size, d));
        let mut v = x;
        for (_, d, own) in around {
            let preds = &self.x.p.blocks[j].preds;
            let n = preds.iter().position(|&p| p == d).ok_or("an if joined past its branch")?;
            let old = match (self.vals.get(&phi.args[n]).copied(), phi.reg == Reg::Scc) {
                (Some(Val::I32(o)), false) | (Some(Val::Bit(o)), true) => o,
                _ => return Err("a group-wide if joining a value with no DXIL form"),
            };
            v = self.s.m.select(at, own, v, old);
        }
        Ok(v)
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

/// A console wave's flag in a loop the whole group takes, at its header: a
/// phi of what each way in brings, and what the round carries back.
pub(super) struct Act {
    phi: V,
    incoming: Vec<(V, usize)>,
    /// This console wave's decision to go round again ([`Lower::group_round`]).
    carry: Option<V>,
}

impl Lower<'_> {
    /// The loops the whole group takes ([`group_loop`]), found once.
    fn find_group_loops(&mut self) {
        let Some(h) = self.halves.as_mut().filter(|h| h.waves > 1 && h.loops.is_none()) else {
            return;
        };
        let lds = h.lds;
        let apart = h
            .apart
            .get_or_insert_with(|| apart_branches(self.x.p, self.x.st, lds).unwrap_or_default());
        let mut loops = HashMap::new();
        for d in self.x.st.wide.iter().chain(apart.iter()).copied() {
            // (a loop the group cannot take kept the program at 64-lane
            // waves, [`several_keep_64`])
            if let Ok(Some(lp)) = group_loop(self.x.p, self.x.st, d) {
                loops.insert(d, lp);
            }
        }
        h.loops = Some(loops);
    }

    /// At the start of IR block `b`, or on a way into it lowered before it:
    /// where `b` begins the rounds of a loop the whole group takes, this
    /// console wave's flag in it, a phi in `b`'s DXIL block, becomes the flag
    /// of the loop's blocks. Whether `b` begins such a loop.
    pub(in super::super) fn group_loop_enter(&mut self, b: usize) -> bool {
        self.find_group_loops();
        if self.halves.as_ref().is_some_and(|h| h.acts.contains_key(&b)) {
            return true;
        }
        let blocks =
            self.halves.as_ref().and_then(|h| h.loops.as_ref()).and_then(|loops| {
                loops.values().find(|lp| lp.header == b).map(|lp| lp.blocks.clone())
            });
        let (Some(blocks), Some(blk)) = (blocks, self.block.get(b).copied().flatten()) else {
            return false;
        };
        let phi = self.s.m.phi(blk, self.s.t.i1);
        let Some(h) = self.halves.as_mut() else { return false };
        for (inner, &inside) in blocks.iter().enumerate() {
            if inside {
                h.flags.insert(inner, phi);
            }
        }
        h.acts.insert(b, Act { phi, incoming: Vec::new(), carry: None });
        true
    }

    /// The branch closing block `b`, the way out of a loop the whole group
    /// takes: the group goes round again while some console wave would. A
    /// console wave already out votes to leave, and its decision, carried to
    /// the next round, keeps it out of the rounds after. The condition for
    /// the way taken.
    fn group_round(&mut self, b: usize, any: bool, some: V) -> Result<V, Why> {
        let LaneExit::Branch { taken, .. } = self.x.st.exits[b] else {
            return Err("a group-wide branch that is no branch");
        };
        let found = self.halves.as_ref().and_then(|h| h.loops.as_ref()).and_then(|l| l.get(&b));
        let Some((header, stays, after)) =
            found.map(|lp| (lp.header, lp.blocks.get(taken).copied(), lp.after.clone()))
        else {
            return Err("a group-wide loop not found");
        };
        // Going round again is what the vote enters.
        let flip = stays != Some(true);
        let c = self.cur;
        let act = match self.halves.as_ref().and_then(|h| h.flags.get(&b).copied()) {
            Some(f) => f,
            None => self.s.m.bool(true),
        };
        // What a console wave posts that leaves ([`Self::wave_enters`]).
        let leaves = self.s.m.i32(u32::from(any == flip));
        let some = self.s.m.select(c, act, some, leaves);
        let (group, own) = self.group_vote(any, flip, some)?;
        let own = self.s.m.bin(c, Bin::And, own, act);
        let Some(h) = self.halves.as_mut() else {
            return Err("a group-wide branch outside halves");
        };
        for (blk, &past) in after.iter().enumerate() {
            if past {
                h.flags.insert(blk, own);
            }
        }
        let act = h.acts.get_mut(&header).ok_or("a group-wide loop entered past its header")?;
        act.carry = Some(own);
        Ok(if flip { self.not(group) } else { group })
    }

    /// A way from IR block `from` (in DXIL block `via`) into the loop
    /// beginning at `to`: a console wave enters it as it ran `from`.
    pub(in super::super) fn group_loop_entry(&mut self, from: usize, to: usize, via: usize) {
        if !self.group_loop_enter(to) {
            return;
        }
        let Some(h) = self.halves.as_ref() else { return };
        let flag = match h.flags.get(&from).copied() {
            Some(f) => f,
            None => self.s.m.bool(true),
        };
        if let Some(act) = self.halves.as_mut().and_then(|h| h.acts.get_mut(&to)) {
            act.incoming.push((flag, via));
        }
    }

    /// A way back from DXIL block `via` to the loop beginning at `to`: it
    /// carries this console wave's decision to go round again.
    pub(in super::super) fn group_loop_back(&mut self, to: usize, via: usize) -> Result<(), Why> {
        let Some(act) = self.halves.as_mut().and_then(|h| h.acts.get_mut(&to)) else {
            return Ok(());
        };
        let carry = act.carry.ok_or("a group-wide loop going round before its branch")?;
        act.incoming.push((carry, via));
        Ok(())
    }

    /// The flags' phis, once every way into each loop is known.
    pub(in super::super) fn fill_group_loops(&mut self) {
        let Some(h) = self.halves.as_mut() else { return };
        for act in std::mem::take(&mut h.acts).into_values() {
            self.s.m.set_phi(act.phi, act.incoming);
        }
    }
}

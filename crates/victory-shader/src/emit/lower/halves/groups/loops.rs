//! Loops the whole thread group takes ([`group_loop`]): see the parent
//! module.

use super::*;

/// A loop the whole group takes at a branch closing one of its blocks, its
/// only way out ([`group_loop`]).
pub(in crate::emit::lower::halves) struct GroupLoop {
    /// The block every round starts at.
    pub(super) header: usize,
    /// Per IR block: whether it lies in the loop.
    pub(super) blocks: Vec<bool>,
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

/// A console wave's flag in a loop the whole group takes, at its header: a
/// phi of what each way in brings, and what the round carries back.
pub(in crate::emit::lower::halves) struct Act {
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
    pub(in crate::emit::lower) fn group_loop_enter(&mut self, b: usize) -> bool {
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
    pub(super) fn group_round(&mut self, b: usize, any: bool, some: V) -> Result<V, Why> {
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
    pub(in crate::emit::lower) fn group_loop_entry(&mut self, from: usize, to: usize, via: usize) {
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
    pub(in crate::emit::lower) fn group_loop_back(
        &mut self,
        to: usize,
        via: usize,
    ) -> Result<(), Why> {
        let Some(act) = self.halves.as_mut().and_then(|h| h.acts.get_mut(&to)) else {
            return Ok(());
        };
        let carry = act.carry.ok_or("a group-wide loop going round before its branch")?;
        act.incoming.push((carry, via));
        Ok(())
    }

    /// The flags' phis, once every way into each loop is known.
    pub(in crate::emit::lower) fn fill_group_loops(&mut self) {
        let Some(h) = self.halves.as_mut() else { return };
        for act in std::mem::take(&mut h.acts).into_values() {
            self.s.m.set_phi(act.phi, act.incoming);
        }
    }
}

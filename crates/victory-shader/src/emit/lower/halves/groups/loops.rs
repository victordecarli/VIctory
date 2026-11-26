//! Loops the whole thread group takes ([`group_loop`]): see the parent
//! module.
//!
//! A loop of several exits ([`joined_exits`]) is left at any of them, each a
//! branch the group takes. A console wave leaving at one runs, in that same
//! round, the block past it on the way to the loop's join, if any (a tail,
//! [`Lower::group_tail`]): it sees what the round left there. What it then
//! takes at the join is captured in that round ([`Lower::group_leave`]), so
//! the join's phis give each console wave its own way out.

use super::*;

/// A loop the whole group takes at a branch closing one of its blocks
/// ([`group_loop`]).
#[derive(Clone)]
pub(in crate::emit::lower::halves) struct GroupLoop {
    /// The block every round starts at.
    pub(super) header: usize,
    /// Per IR block: whether it lies in the loop.
    pub(super) blocks: Vec<bool>,
    /// Per IR block: in the loop past the branch, before the round ends.
    after: Vec<bool>,
    /// For a loop of several exits, where they join ([`joined_exits`]).
    join: Option<usize>,
    /// Per IR block: a tail on the way from an exit to the join, run in the
    /// round a console wave leaves.
    pub(super) tails: Vec<bool>,
    /// The ways out: each exiting block and the block past it.
    exits: Vec<(usize, usize)>,
}

const LEFT_APART: Why = "a group-wide loop left at more than one place";

/// The loop around the branch closing block `d`, None when `d` is in none:
/// the innermost loop holding `d` that `d` leaves ([`loop_left_at`]). The
/// whole group takes it only where every round passes the branch, so that
/// each round a console wave goes on or leaves there, and where its other
/// exits, if any, are branches the group takes too (`taken`), joining as
/// [`joined_exits`] has it. A loop inside it is taken so too, each with its
/// own flag. `lds`: the program has LDS of its own.
pub(super) fn group_loop(
    p: &Program,
    st: &Structured,
    d: usize,
    taken: &HashSet<usize>,
    lds: bool,
) -> Result<Option<GroupLoop>, Why> {
    let succ = successors(st).ok_or("an indirect exit")?;
    let LaneExit::Branch { taken: way, not_taken, .. } = st.exits[d] else {
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
    let exits: Vec<(usize, usize)> = (0..n)
        .filter(|&b| blocks[b])
        .flat_map(|b| succ[b].iter().filter(|&&t| !blocks[t]).map(move |&t| (b, t)))
        .collect();
    let (join, tails) = match exits.len() {
        1 => (None, vec![false; n]),
        _ => {
            let (j, t) = joined_exits(p, st, &succ, &blocks, &exits, taken, lds)?;
            (Some(j), t)
        }
    };
    let inside = match (blocks[way], blocks[not_taken]) {
        (true, false) => way,
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
    Ok(Some(GroupLoop { header, blocks, after, join, tails, exits }))
}

/// Where the exits of a loop (`blocks`) of several join, and per IR block
/// whether it is a tail: each exit a branch the group takes (`taken`) with
/// one way out, going straight to the join (the block first post-dominating
/// every way out) or to a tail, a block that only those exits lead to,
/// with no place where the halves or the group meet, that jumps to the
/// join. The join is reached from the exits and the tails alone.
fn joined_exits(
    p: &Program,
    st: &Structured,
    succ: &[Vec<usize>],
    blocks: &[bool],
    exits: &[(usize, usize)],
    taken: &HashSet<usize>,
    lds: bool,
) -> Result<(usize, Vec<bool>), Why> {
    let n = succ.len();
    let exiting = |b: usize| exits.iter().filter(|e| e.0 == b).count();
    if exits.iter().any(|&(b, _)| !taken.contains(&b) || exiting(b) != 1) {
        return Err(LEFT_APART);
    }
    let pdom = post_dominators(succ);
    let all = |c: usize| exits.iter().all(|&(_, t)| pdom[t][c]);
    let common: Vec<usize> = (0..n).filter(|&c| all(c)).collect();
    let join =
        common.iter().copied().find(|&c| common.iter().all(|&o| pdom[c][o])).ok_or(LEFT_APART)?;
    let mut tails = vec![false; n];
    for &(_, t) in exits.iter().filter(|e| e.1 != join) {
        let only_exits = p.blocks[t].preds.iter().all(|q| exits.contains(&(*q, t)));
        if st.exits[t] != LaneExit::Jump(join) || !only_exits || meets_in(p, st, t, lds) {
            return Err(LEFT_APART);
        }
        tails[t] = true;
    }
    let reached = p.blocks[join].preds.iter().all(|&q| tails[q] || exits.contains(&(q, join)));
    if blocks[join] || !reached {
        return Err(LEFT_APART);
    }
    Ok((join, tails))
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
    /// For a loop of several exits: how each console wave left.
    exits: Option<Exits>,
}

/// How each console wave left a loop of several exits ([`Lower::group_leave`]).
struct Exits {
    /// Where the exits join.
    join: usize,
    /// Whether this console wave left already.
    left: Carried,
    /// Per phi of the join, when some exit goes through a tail: what this
    /// console wave takes there, as of the round it left.
    caps: Vec<Option<Carried>>,
    /// Per way out the group can take (exiting block, block past it): what
    /// each phi of the join takes over it.
    out: Vec<(usize, usize, Vec<Option<V>>)>,
}

/// A value carried round a loop: a phi at its header, the phi's incoming
/// values, and the value so far in the round.
struct Carried {
    phi: V,
    incoming: Vec<(V, usize)>,
    now: V,
}

impl Carried {
    fn new(phi: V) -> Self {
        Carried { phi, incoming: Vec::new(), now: phi }
    }
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
        let taken: HashSet<usize> = self.x.st.wide.iter().chain(apart.iter()).copied().collect();
        let mut loops = HashMap::new();
        for &d in &taken {
            // (a loop the group cannot take kept the program at 64-lane
            // waves, [`several_keep_64`])
            if let Ok(Some(lp)) = group_loop(self.x.p, self.x.st, d, &taken, lds) {
                loops.insert(d, lp);
            }
        }
        h.loops = Some(loops);
    }

    /// Whether IR block `b` is a tail of a loop of several exits the whole
    /// group takes: it has no DXIL block of its own, a console wave runs it
    /// in the round it leaves ([`Self::group_tail`]).
    pub(in crate::emit::lower) fn group_tail_block(&mut self, b: usize) -> bool {
        self.find_group_loops();
        let loops = self.halves.as_ref().and_then(|h| h.loops.as_ref());
        loops.is_some_and(|l| l.values().any(|lp| lp.tails.get(b).copied().unwrap_or(false)))
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
        let found = self.halves.as_ref().and_then(|h| h.loops.as_ref()).and_then(|loops| {
            loops
                .values()
                .find(|lp| lp.header == b)
                .map(|lp| (lp.blocks.clone(), lp.join, lp.tails.contains(&true)))
        });
        let (Some((blocks, join, tails)), Some(blk)) =
            (found, self.block.get(b).copied().flatten())
        else {
            return false;
        };
        let phi = self.s.m.phi(blk, self.s.t.i1);
        let exits = join.map(|join| {
            let left = self.s.m.phi(blk, self.s.t.i1);
            let caps = self.x.p.blocks[join]
                .phis
                .iter()
                .map(|jp| {
                    let ty = match jp.reg {
                        Reg::S(c) if c == EXEC || c == EXEC + 1 => return None,
                        Reg::Scc => self.s.t.i1,
                        Reg::S(_) | Reg::V(_) => self.s.t.i32,
                    };
                    tails.then(|| Carried::new(self.s.m.phi(blk, ty)))
                })
                .collect();
            Exits { join, left: Carried::new(left), caps, out: Vec::new() }
        });
        let Some(h) = self.halves.as_mut() else {
            return false;
        };
        for (inner, &inside) in blocks.iter().enumerate() {
            if inside {
                h.flags.insert(inner, phi);
            }
        }
        h.acts.insert(b, Act { phi, incoming: Vec::new(), carry: None, exits });
        true
    }

    /// The branch closing block `b`, a way out of a loop the whole group
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

    /// Past the vote at block `e` ([`Self::group_round`]), an exit of a loop
    /// of several exits: a console wave that leaves there now runs the tail
    /// past it, if any ([`Self::group_tail`]), and what it takes at the
    /// join, over this way, is captured; a wave that left in an earlier
    /// round keeps what it took then, and one that never ran the loop (its
    /// flag off from the way in) takes what this way brings, its VGPRs as
    /// they were.
    pub(in crate::emit::lower) fn group_leave(&mut self, e: usize) -> Result<(), Stop> {
        let p = self.x.p;
        let found = self.halves.as_ref().and_then(|h| h.loops.as_ref()).and_then(|l| l.get(&e));
        let Some(lp) = found.filter(|lp| lp.join.is_some()).cloned() else {
            return Ok(());
        };
        let Some(&(_, t)) = lp.exits.iter().find(|x| x.0 == e) else {
            return Err(internal(format!("block {e} is no exit of its loop")));
        };
        let h = self
            .halves
            .as_ref()
            .ok_or_else(|| internal("a group-wide loop outside halves".into()))?;
        let own = h.acts.get(&lp.header).and_then(|a| a.carry);
        let act = h.flags.get(&e).copied();
        let Some(own) = own else {
            return Err(internal("a group-wide loop left before its vote".into()));
        };
        let c = self.cur;
        let act = match act {
            Some(f) => f,
            None => self.s.m.bool(true),
        };
        let t1 = self.s.m.bool(true);
        let stays_not = self.s.m.bin(c, Bin::Xor, own, t1);
        let leaves = self.s.m.bin(c, Bin::And, act, stays_not);
        let via = if lp.tails[t] {
            self.group_tail(e, t, leaves)?;
            t
        } else {
            e
        };
        let Some(join) = lp.join else { return Ok(()) };
        let n = p.blocks[join].preds.iter().position(|&q| q == via).ok_or_else(|| {
            internal(format!("an exit's way {via} not into its loop's join {join}"))
        })?;
        let c = self.cur;
        let Some(ex) = self
            .halves
            .as_mut()
            .and_then(|h| h.acts.get_mut(&lp.header))
            .and_then(|a| a.exits.take())
        else {
            return Err(internal("a loop of several exits without its record".into()));
        };
        let mut ex = ex;
        let left = self.s.m.bin(c, Bin::Or, ex.left.now, leaves);
        let mut values = Vec::with_capacity(ex.caps.len());
        for (k, jp) in p.blocks[join].phis.iter().enumerate() {
            let x = match (self.vals.get(&jp.args[n]).copied(), jp.reg) {
                (_, Reg::S(r)) if r == EXEC || r == EXEC + 1 => None,
                (Some(Val::Bit(x)), Reg::Scc) => Some(x),
                (Some(Val::I32(x)), Reg::S(_) | Reg::V(_)) => Some(x),
                _ => None,
            };
            let v = match (x, ex.caps.get_mut(k).and_then(Option::as_mut)) {
                (Some(x), Some(cap)) => {
                    let now = self.s.m.select(c, leaves, x, cap.now);
                    cap.now = now;
                    Some(self.s.m.select(c, left, now, x))
                }
                (x, _) => x,
            };
            values.push(v);
        }
        ex.left.now = left;
        ex.out.push((e, t, values));
        if let Some(a) = self.halves.as_mut().and_then(|h| h.acts.get_mut(&lp.header)) {
            a.exits = Some(ex);
        }
        Ok(())
    }

    /// Tail `t` of a loop of several exits ([`joined_exits`]), run past the
    /// exit at block `e` in the round a console wave leaves there: in `e`'s
    /// DXIL block, its phis taking what the way from `e` brings, `flag` (the
    /// console wave leaves now) guarding it.
    fn group_tail(&mut self, e: usize, t: usize, flag: V) -> Result<(), Stop> {
        let p = self.x.p;
        let n = p.blocks[t]
            .preds
            .iter()
            .position(|&q| q == e)
            .ok_or_else(|| internal(format!("tail {t} not reached from exit {e}")))?;
        for phi in &p.blocks[t].phis {
            let arg = phi.args[n];
            let v = match phi.reg {
                Reg::S(c) if c == EXEC || c == EXEC + 1 => Val::Missing("a lane mask"),
                Reg::S(_) | Reg::Scc | Reg::V(_) => self
                    .vals
                    .get(&arg)
                    .copied()
                    .unwrap_or(Val::Missing("a tail's join of a value with no DXIL form")),
            };
            self.vals.insert(phi.value, v);
            if let Some(&m) = self.x.st.masks.get(&arg) {
                let lanes =
                    self.boolean(m, self.cur).map_err(|w| unsupported(p.blocks[t].offset, w))?;
                self.mask_phis.insert(phi.value, lanes);
            }
        }
        if let Some(h) = self.halves.as_mut() {
            h.flags.insert(t, flag);
        }
        let ir = std::mem::replace(&mut self.cur_ir, t);
        for i in 0..p.blocks[t].insts.len() {
            self.inst(t, i)?;
        }
        self.cur_ir = ir;
        Ok(())
    }

    /// The DXIL block the way from IR block `from` to `to` goes to, where it
    /// leaves a loop of several exits the whole group takes: the join (its
    /// tail, if any, ran in the exiting block, [`Self::group_leave`]).
    pub(in crate::emit::lower) fn group_exit_target(
        &self,
        from: usize,
        to: usize,
    ) -> Option<usize> {
        let loops = self.halves.as_ref()?.loops.as_ref()?;
        let lp = loops.get(&from).filter(|lp| lp.exits.contains(&(from, to)))?;
        self.block.get(lp.join?).copied().flatten()
    }

    /// The incoming values of phi `i` of IR block `j`, where `j` joins the
    /// exits of a loop the whole group takes: over each way out the group can
    /// take, what [`Self::group_leave`] captured. None elsewhere.
    pub(in crate::emit::lower) fn group_exits_join(
        &self,
        j: usize,
        i: usize,
    ) -> Option<Vec<(Option<V>, usize)>> {
        let ex = self
            .halves
            .as_ref()?
            .acts
            .values()
            .find_map(|a| a.exits.as_ref().filter(|x| x.join == j))?;
        Some(
            ex.out
                .iter()
                .filter_map(|(e, t, values)| {
                    let at = self.edges.get(&(*e, *t)).copied()?;
                    Some((values.get(i).copied().flatten(), at))
                })
                .collect(),
        )
    }

    /// A way from IR block `from` (in DXIL block `via`) into the loop
    /// beginning at `to`: a console wave enters it as it ran `from`.
    pub(in crate::emit::lower) fn group_loop_entry(&mut self, from: usize, to: usize, via: usize) {
        if !self.group_loop_enter(to) {
            return;
        }
        let Some(h) = self.halves.as_ref() else {
            return;
        };
        let flag = match h.flags.get(&from).copied() {
            Some(f) => f,
            None => self.s.m.bool(true),
        };
        let (no, zero, nothing) = (self.s.m.bool(false), self.s.m.i32(0), self.s.m.bool(false));
        if let Some(act) = self.halves.as_mut().and_then(|h| h.acts.get_mut(&to)) {
            act.incoming.push((flag, via));
            if let Some(ex) = act.exits.as_mut() {
                ex.left.incoming.push((no, via));
                for cap in ex.caps.iter_mut().flatten() {
                    let bit = self.s.m.type_of(cap.phi) == self.s.t.i1;
                    cap.incoming.push((if bit { nothing } else { zero }, via));
                }
            }
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
        if let Some(ex) = act.exits.as_mut() {
            ex.left.incoming.push((ex.left.now, via));
            for cap in ex.caps.iter_mut().flatten() {
                cap.incoming.push((cap.now, via));
            }
        }
        Ok(())
    }

    /// The flags' phis, once every way into each loop is known.
    pub(in crate::emit::lower) fn fill_group_loops(&mut self) {
        let Some(h) = self.halves.as_mut() else {
            return;
        };
        for act in h.acts.values_mut() {
            self.s.m.set_phi(act.phi, std::mem::take(&mut act.incoming));
            if let Some(ex) = act.exits.as_mut() {
                self.s.m.set_phi(ex.left.phi, std::mem::take(&mut ex.left.incoming));
                for cap in ex.caps.iter_mut().flatten() {
                    self.s.m.set_phi(cap.phi, std::mem::take(&mut cap.incoming));
                }
            }
        }
    }
}

//! Control flow: the order blocks are lowered in, their exits (branches,
//! loop back edges counted against the iteration limit, the program's end)
//! and the phis filled once every block is lowered.

use super::*;

impl Lower<'_> {
    pub(super) fn run(mut self) -> Result<Lowered, Stop> {
        let p = self.x.p;
        let order = crate::ir::reverse_postorder(&p.blocks);
        for (n, &b) in order.iter().enumerate() {
            self.rpo[b] = n;
        }
        self.place_blocks(&order);
        self.cur = self.s.entry;
        self.find_sections();
        self.inputs()?;
        if let Some(mesh) = self.s.mesh {
            self.mesh_prologue(mesh).map_err(|w| unsupported(0, w))?;
            self.block[order[0]] = Some(self.cur);
        }
        for &b in &order {
            if self.block[b].is_some() {
                self.lower_block(b)?;
            }
        }
        self.fill_phis()?;
        self.finish()
    }

    /// A DXIL block for each IR block in reverse postorder (the entry
    /// first), and an iteration counter for each loop header
    /// (`LOOP_LIMIT`).
    fn place_blocks(&mut self, order: &[usize]) {
        let p = self.x.p;
        let mut blocks = Vec::with_capacity(order.len());
        for (n, &b) in order.iter().enumerate() {
            // (a tail of a loop of several exits runs in its exiting blocks)
            if n > 0 && self.group_tail_block(b) {
                continue;
            }
            let blk = if n == 0 { self.s.entry } else { self.s.m.new_block() };
            self.block[b] = Some(blk);
            blocks.push((b, blk));
        }
        // Loop headers count iterations (`LOOP_LIMIT`).
        for &(b, blk) in &blocks {
            let header = p.blocks[b]
                .preds
                .iter()
                .any(|&q| self.block[q].is_some() && self.rpo[b] <= self.rpo[q]);
            if header {
                let phi = self.s.m.phi(blk, self.s.t.i32);
                self.counters.insert(b, (phi, Vec::new()));
            }
        }
    }

    /// IR block `b`: its phis, its instructions (a whole-wave section as
    /// one) and its exit.
    fn lower_block(&mut self, b: usize) -> Result<(), Stop> {
        let p = self.x.p;
        self.cur = self.dxil_block(b)?;
        self.cur_ir = b;
        // (the flag's phi, if `b` begins a loop the group takes)
        self.group_loop_enter(b);
        let preds: Vec<usize> =
            p.blocks[b].preds.iter().copied().filter(|&q| self.block[q].is_some()).collect();
        self.exported[b] = !preds.is_empty() && preds.iter().all(|&q| self.exported_out(q));
        for (i, phi) in p.blocks[b].phis.iter().enumerate() {
            let val = match phi.reg {
                Reg::S(c) if c == EXEC || c == EXEC + 1 => Val::Missing("a lane mask"),
                _ if self.bad.contains_key(&phi.value) => Val::Missing(self.bad[&phi.value]),
                Reg::Scc => {
                    let v = self.s.m.phi(self.cur, self.s.t.i1);
                    self.data_phis.push((v, b, i));
                    Val::Bit(v)
                }
                _ => {
                    let v = self.s.m.phi(self.cur, self.s.t.i32);
                    self.data_phis.push((v, b, i));
                    Val::I32(v)
                }
            };
            self.vals.insert(phi.value, val);
        }
        let mut i = 0;
        while i < p.blocks[b].insts.len() {
            self.inst(b, i)?;
            if let Some(end) = self.sections.get(&b).and_then(|s| s.get(&i)).map(|s| s.end) {
                for j in i + 1..end {
                    self.section_inst(b, j, i);
                }
                i = end;
                continue;
            }
            i += 1;
        }
        self.mask_ok = self.mask_data == MaskData::Console;
        self.tail[b] = Some(self.cur);
        self.exit(b)?;
        // (a loop's tail may have run in it, [`Self::group_leave`])
        self.tail[b] = Some(self.cur);
        Ok(())
    }

    /// What the lowering hands back once every block is lowered.
    fn finish(mut self) -> Result<Lowered, Stop> {
        let user_data = self.user_reads.values().copied().collect();
        self.s.samplers = self.samplers.len() as u32;
        let root_cbvs = self.s.root_cbvs.iter().map(|r| r.0).collect();
        let root_srvs = self.s.root_srvs.iter().map(|r| r.0).collect();
        let cmp_samplers = self.s.cmp_samplers;
        Ok(Lowered {
            cmp_samplers,
            root_cbvs,
            root_srvs,
            #[cfg(test)]
            nonuniform_handles: self.s.nonuniform_handles,
            #[cfg(test)]
            wave_ops: self.s.flags & flags::WAVE_OPS != 0,
            #[cfg(test)]
            barriers: self.s.barriers,
            #[cfg(test)]
            lane_reads: self.s.lane_reads,
            #[cfg(test)]
            flagged_accesses: self.flagged_accesses,
            #[cfg(test)]
            group_votes: self.group_votes,
            wave_size: self.s.wave_size,
            dxil: self.s.finish().map_err(Stop::Fail)?,
            user_data,
            samplers: self.samplers,
        })
    }

    fn exported_out(&self, b: usize) -> bool {
        self.exported[b] || self.has_done_export(b)
    }

    fn has_done_export(&self, b: usize) -> bool {
        (0..self.x.p.blocks[b].insts.len())
            .any(|i| self.exports.get(&(b, i)).is_some_and(|e| e.done))
    }

    fn exit(&mut self, b: usize) -> Result<(), Stop> {
        let offset =
            self.x.p.blocks[b].insts.last().map_or(self.x.p.blocks[b].offset, |i| i.machine.offset);
        let term = match self.x.st.exits[b] {
            LaneExit::Jump(t) if self.counters.contains_key(&t) && self.rpo[t] <= self.rpo[b] => {
                let ok = self.count_iteration(t, pred::IULT)?;
                let abort = self.abort_block();
                Term::CondBr(ok, self.dxil_block(t)?, abort)
            }
            LaneExit::Jump(t) => {
                self.enter_loop(b, t, self.cur);
                // (into a block both ways of a group-wide if share: it runs here)
                self.group_shared(b, t)?;
                match self.group_target(b, t) {
                    Some(to) => Term::Br(to),
                    None => Term::Br(self.dxil_block(t)?),
                }
            }
            LaneExit::Branch { when, taken, not_taken } => {
                // A negated condition branches the other way round.
                let (cond, taken, not_taken) = match self.x.st.bools.node(when) {
                    _ if self.x.st.wide.contains(&b) => {
                        let c = self.wave_wide(b, when).map_err(|w| unsupported(offset, w))?;
                        (c, taken, not_taken)
                    }
                    _ if self.group_apart(b) => {
                        let c = self.group_scalar(b, when).map_err(|w| unsupported(offset, w))?;
                        (c, taken, not_taken)
                    }
                    Some((atom, Bool::TRUE, Bool::FALSE)) => {
                        let a = self.atom(atom, self.cur).map_err(|w| unsupported(offset, w))?;
                        (a, not_taken, taken)
                    }
                    _ => {
                        let c = self.boolean(when, self.cur).map_err(|w| unsupported(offset, w))?;
                        (c, taken, not_taken)
                    }
                };
                // (past a vote at an exit of a loop of several exits)
                self.group_leave(b)?;
                // A back edge is taken only below the iteration limit.
                let back =
                    |l: &Self, to: usize| l.counters.contains_key(&to) && l.rpo[to] <= l.rpo[b];
                // (No `xor` of a compare: a loop exit testing one hung the
                // AMD driver.)
                let cond = if back(self, taken) {
                    let ok = self.count_iteration(taken, pred::IULT)?;
                    self.s.m.bin(self.cur, Bin::And, cond, ok)
                } else if back(self, not_taken) {
                    let stop = self.count_iteration(not_taken, pred::IUGE)?;
                    self.s.m.bin(self.cur, Bin::Or, cond, stop)
                } else {
                    cond
                };
                // Critical edges get a block of their own, so loops have
                // dedicated exits and preheaders as DXC emits them.
                let (t, f) = (self.edge(b, taken)?, self.edge(b, not_taken)?);
                for (to, via) in [(taken, t), (not_taken, f)] {
                    if !back(self, to) {
                        // From an edge block, or from here.
                        let from = if Some(via) == self.block[to] { self.cur } else { via };
                        self.enter_loop(b, to, from);
                    }
                }
                Term::CondBr(cond, t, f)
            }
            LaneExit::End => {
                if self.s.pixel && !self.exported[b] && !self.has_done_export(b) {
                    // A lane leaving without its final export was killed.
                    let t = self.s.m.bool(true);
                    self.s.op(self.cur, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
                }
                Term::Ret
            }
            LaneExit::Indirect => return Err(unsupported(offset, "an indirect jump")),
        };
        let _ = Exit::End;
        self.s.m.terminate(self.cur, term);
        Ok(())
    }

    /// A forward edge into loop header `to` (from DXIL block `via`) starts
    /// its count at 0.
    fn enter_loop(&mut self, from: usize, to: usize, via: usize) {
        if let Some((_, incoming)) = self.counters.get_mut(&to) {
            let z = self.s.m.i32(0);
            incoming.push((z, via));
        }
        self.group_loop_entry(from, to, via);
    }

    /// A back edge to header `to` (a back edge keeps its block, no
    /// critical-edge block): the count goes up; the result compares it with
    /// [`LOOP_LIMIT`] by `p` (below: go on; at or above: stop).
    fn count_iteration(&mut self, to: usize, p: u8) -> Result<V, Stop> {
        let c = self.cur;
        let Some((phi, incoming)) = self.counters.get_mut(&to) else {
            return Err(internal(format!("a back edge to block {to}, not a loop header")));
        };
        let one = self.s.m.i32(1);
        let next = self.s.m.bin(c, Bin::Add, *phi, one);
        let limit = self.s.m.i32(super::super::loop_limit());
        incoming.push((next, c));
        self.group_loop_back(to, c).map_err(|w| internal(w.to_string()))?;
        Ok(self.s.m.cmp(c, p, next, limit))
    }

    /// The block a lane stopped by [`LOOP_LIMIT`] leaves through.
    fn abort_block(&mut self) -> usize {
        if let Some(a) = self.abort {
            return a;
        }
        let a = self.s.m.new_block();
        if self.s.pixel {
            let t = self.s.m.bool(true);
            self.s.op(a, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
        }
        self.s.m.terminate(a, Term::Ret);
        self.abort = Some(a);
        a
    }

    /// The DXIL block of IR block `b`. Every block reachable from the entry
    /// has one (`run` makes them first), and only those are visited.
    fn dxil_block(&self, b: usize) -> Result<usize, Stop> {
        self.block
            .get(b)
            .copied()
            .flatten()
            .ok_or_else(|| internal(format!("IR block {b} reached but given no DXIL block")))
    }

    /// The DXIL block a branch from IR block `from` to `to` goes to.
    fn edge(&mut self, from: usize, to: usize) -> Result<usize, Stop> {
        // An exit of a loop of several exits gets a block of its own.
        if let Some(join) = self.group_exit_target(from, to) {
            let e = self.s.m.new_block();
            self.s.m.terminate(e, Term::Br(join));
            self.edges.insert((from, to), e);
            return Ok(e);
        }
        let target = match self.group_target(from, to) {
            Some(t) => t,
            None => self.dxil_block(to)?,
        };
        let preds = self.x.p.blocks[to].preds.iter().filter(|&&q| self.block[q].is_some()).count();
        // Back edges stay as they are: the exiting block is the latch.
        if preds < 2 || self.rpo[to] <= self.rpo[from] {
            return Ok(target);
        }
        let e = self.s.m.new_block();
        self.s.m.terminate(e, Term::Br(target));
        self.edges.insert((from, to), e);
        Ok(e)
    }

    fn fill_phis(&mut self) -> Result<(), Stop> {
        self.close_read_phis();
        for (v, b, i) in std::mem::take(&mut self.data_phis) {
            self.fill_data_phi(v, b, i)?;
        }
        for (phi, incoming) in std::mem::take(&mut self.counters).into_values() {
            self.s.m.set_phi(phi, incoming);
        }
        self.fill_group_loops();
        while let Some((v, b, i)) = self.pending_masks.pop() {
            self.fill_mask_phi(v, b, i)?;
        }
        Ok(())
    }

    /// Marks read the phis that only read phis consume.
    fn close_read_phis(&mut self) {
        let p = self.x.p;
        // A phi that only another read phi consumes is live too. Without this, a value with no DXIL form that
        // reaches a read through two joins is taken for dead at the first join and replaced by zero below, and the
        // lowering accepts what it refuses when the same value is read directly.
        let mut work: Vec<Value> = self.read_phis.iter().copied().collect();
        while let Some(v) = work.pop() {
            let Some(&(b, i)) = self.phi_at.get(&v) else {
                continue;
            };
            for &arg in &p.blocks[b].phis[i].args {
                if self.phi_at.contains_key(&arg) && self.read_phis.insert(arg) {
                    work.push(arg);
                }
            }
        }
    }

    /// Data phi `v` of IR phi `i` in block `b` (an SGPR, VGPR or SCC): a
    /// value with no DXIL form fails a phi that is read, and is zero in
    /// one that is not.
    fn fill_data_phi(&mut self, v: V, b: usize, i: usize) -> Result<(), Stop> {
        let p = self.x.p;
        let phi = &p.blocks[b].phis[i];
        let bit = phi.reg == Reg::Scc;
        if let Some(ways) = self.group_exits_join(b, i) {
            let mut incoming = Vec::with_capacity(ways.len());
            for (x, at) in ways {
                let x = match x {
                    Some(x) => x,
                    None if self.read_phis.contains(&phi.value) => {
                        let why = "a loop's exits joining a value with no DXIL form";
                        return Err(Stop::Retry(phi.value, why));
                    }
                    None if bit => self.s.m.bool(false),
                    None => self.s.m.i32(0),
                };
                incoming.push((x, at));
            }
            self.s.m.set_phi(v, incoming);
            return Ok(());
        }
        let mut incoming = Vec::new();
        for (n, &q) in p.blocks[b].preds.iter().enumerate() {
            let Some(qb) = self.tail[q] else { continue };
            if self.group_copied(q, b) {
                continue;
            }
            let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
            let mut ways = vec![(self.vals.get(&phi.args[n]).copied(), qb, q)];
            // (past a block both ways of a group-wide if share, its runs in the region)
            ways.extend(self.group_copies(b, i, q));
            for (val, qb, q) in ways {
                let x = match (val, bit) {
                    (Some(Val::I32(x)), false) | (Some(Val::Bit(x)), true) => x,
                    (Some(Val::Bit(_)), false) if self.read_phis.contains(&phi.value) => {
                        return Err(Stop::Retry(phi.value, "an SCC joined with SGPR values"));
                    }
                    (Some(Val::Missing(why)), _) if self.read_phis.contains(&phi.value) => {
                        return Err(Stop::Retry(phi.value, why));
                    }
                    _ if self.read_phis.contains(&phi.value) => {
                        return Err(Stop::Retry(phi.value, "a join of mismatched values"));
                    }
                    _ if bit => self.s.m.bool(false),
                    _ => self.s.m.i32(0),
                };
                let x = if self.read_phis.contains(&phi.value) {
                    self.group_join(b, i, q, qb, x).map_err(|w| Stop::Retry(phi.value, w))?
                } else {
                    x
                };
                incoming.push((x, qb, q));
            }
        }
        let zero = if bit { self.s.m.bool(false) } else { self.s.m.i32(0) };
        let incoming = self.group_arms(b, incoming, zero);
        self.s.m.set_phi(v, incoming);
        Ok(())
    }

    /// Lane-mask phi `v` of IR phi `i` in block `b`: each way's boolean.
    fn fill_mask_phi(&mut self, v: V, b: usize, i: usize) -> Result<(), Stop> {
        let p = self.x.p;
        let phi = &p.blocks[b].phis[i];
        let offset = p.blocks[b].offset;
        if self.group_exits_join(b, i).is_some() {
            return Err(unsupported(offset, "a lane-mask join past a loop of several exits"));
        }
        let mut incoming = Vec::new();
        for (n, &q) in p.blocks[b].preds.iter().enumerate() {
            let Some(qb) = self.tail[q] else { continue };
            if self.group_copied(q, b) {
                continue;
            }
            if !self.group_copies(b, i, q).is_empty() {
                let why = "a lane-mask join past a block both ways of a group-wide if share";
                return Err(unsupported(offset, why));
            }
            let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
            let m = *self
                .x
                .st
                .masks
                .get(&phi.args[n])
                .ok_or_else(|| unsupported(offset, "a lane-mask join of an untracked mask"))?;
            let x = self.boolean(m, qb).map_err(|w| unsupported(offset, w))?;
            incoming.push((x, qb, q));
        }
        let zero = self.s.m.bool(false);
        let incoming = self.group_arms(b, incoming, zero);
        self.s.m.set_phi(v, incoming);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{S_ENDPGM, lowers};

    #[test]
    fn a_missing_value_stays_missing_through_two_live_joins() {
        let mut words = [
            0xF490_0180,
            0,           // s_memtime s[6:7]
            0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
            0xBF85_0001, // skip the first overwrite
            0xBE86_0380, // s_mov_b32 s6, 0
            0xBF06_8101, // s_cmp_eq_u32 s1, 1 (s1 = group id Y)
            0xBF85_0001, // skip the second overwrite
            0xBE86_0381, // s_mov_b32 s6, 1
            0xBF07_8006, // s_cmp_lg_u32 s6, 0
            0xBF85_0001, // use the result to branch
            0xBE83_0380,
            S_ENDPGM,
        ];
        let group_ids_xy = 3 << 7;
        // Compare to a direct read so future timestamp support works in both cases.
        let refusal = |code: &[u32]| match lowers(code, group_ids_xy) {
            Ok(()) => None,
            Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
            Err(error) => panic!("{error}"),
        };
        let direct = [&words[..2], &words[8..]].concat();
        assert_eq!(refusal(&words), refusal(&direct));
        words[8] = 0xBF07_8000; // read the group id instead; the timestamp is dead
        lowers(&words, group_ids_xy).expect("dead timestamp");
    }
}

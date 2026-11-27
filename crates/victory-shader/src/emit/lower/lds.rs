//! LDS (thread-group shared memory): reads, writes and atomics, and
//! `ds_swizzle`, which moves values between lanes.

use super::super::bitcode::Rmw;
use super::*;

/// Dwords per element of a `ds_read`/`ds_write` by its name's suffix.
fn width(name: &str) -> u32 {
    match name.rsplit('_').next() {
        Some("b64") => 2,
        Some("b96") => 3,
        Some("b128") => 4,
        _ => 1,
    }
}

/// (byte offset, dwords) per element of a read or write named `name` with
/// the part after `ds_read`/`ds_write` in `rest`; `None` for bytes and
/// halves. read2/write2 scale their two offsets by the element size (×64
/// for st64).
fn elements(name: &str, rest: &str, offset0: u8, offset1: u8) -> Option<Vec<(u32, u32)>> {
    if rest.starts_with('2') {
        let w = width(name);
        let scale = 4 * w * if name.contains("st64") { 64 } else { 1 };
        Some(vec![(u32::from(offset0) * scale, w), (u32::from(offset1) * scale, w)])
    } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
        let offset = u32::from(offset0) | u32::from(offset1) << 8;
        Some(vec![(offset, width(name))])
    } else {
        None
    }
}

/// The read-modify-write of an LDS atomic named `op` (without `ds_` and
/// `_rtn`); `None` for compare-and-store.
fn rmw(op: &str) -> Result<Option<Rmw>, Why> {
    Ok(match op {
        "add_u32" => Some(Rmw::Add),
        "sub_u32" => Some(Rmw::Sub),
        "and_b32" => Some(Rmw::And),
        "or_b32" => Some(Rmw::Or),
        "xor_b32" => Some(Rmw::Xor),
        "min_i32" => Some(Rmw::Min),
        "max_i32" => Some(Rmw::Max),
        "min_u32" => Some(Rmw::UMin),
        "max_u32" => Some(Rmw::UMax),
        "wrxchg_b32" => Some(Rmw::Xchg),
        "cmpst_b32" => None,
        _ => return Err("an LDS operation"),
    })
}

impl Lower<'_> {
    /// LDS (thread-group shared memory) and `ds_swizzle`.
    pub(super) fn ds(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Ds { dst, addr, data, offset0, offset1, gds } = &inst.fields else {
            return Err("an LDS instruction without DS fields");
        };
        if *gds {
            return Err("GDS");
        }
        let name = inst.name();
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let offset = u32::from(*offset0) | u32::from(*offset1) << 8;
        let dst_reg = match dst {
            Some(Operand::Vector { reg, .. }) => Some(*reg),
            _ => None,
        };
        if name == "ds_swizzle_b32" {
            let r = self.swizzle(inst, addr.as_ref(), offset, c)?;
            let reg = dst_reg.ok_or("a swizzle result")?;
            self.write(inst, Reg::V(reg), r, guard);
            return Ok(());
        }
        // The halves' own dwords are not the program's.
        let reserved = self.halves.as_ref().map_or(0, halves::Halves::dwords);
        if self.s.lds_dwords == reserved {
            return Err("LDS the program does not allocate");
        }
        if self.halves.as_ref().is_some_and(halves::Halves::orders_lds) {
            // The console wave's LDS order across its two host waves.
            self.group_sync(self.cur);
        }
        let a = self.vgprs(inst, addr.as_ref().ok_or("an LDS address")?, 1)?[0];
        if let Some(rest) = name.strip_prefix("ds_read") {
            let elements =
                elements(name, rest, *offset0, *offset1).ok_or("an LDS read of bytes or halves")?;
            let reg = dst_reg.ok_or("an LDS read result")?;
            self.lds_read(inst, a, &elements, reg, guard);
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("ds_write") {
            let elements = elements(name, rest, *offset0, *offset1)
                .ok_or("an LDS write of bytes or halves")?;
            return self.lds_write(inst, a, elements, data, guard);
        }
        self.lds_atomic(inst, a, offset, data, dst_reg, guard)
    }

    /// `ds_swizzle_b32` of `src` with pattern `offset`. Bit 15 set: each
    /// lane of a quad picks a lane of it; else lanes of each group of 32
    /// combine their index with and/or/xor masks. The source sits in the
    /// address field.
    fn swizzle(
        &mut self,
        inst: &Inst,
        addr: Option<&Operand>,
        offset: u32,
        c: usize,
    ) -> Result<V, Why> {
        let quad = offset & 0x8000 != 0
            || offset & 0x1C == 0x1C && offset & 0x380 == 0 && offset >> 10 & 0x1F < 4;
        // Mesh: host quads are threads 4n..4n+3 of a 1D group, as the
        // console's, so quad-local swizzles read through quad operations.
        let mesh_quad = self.s.mesh.is_some() && quad;
        if !mesh_quad {
            self.cross_lane(quad)?;
        }
        let x = self.vgprs(inst, addr.ok_or("a swizzle source")?, 1)?[0];
        let lane = self.lane_index();
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let src = if offset & 0x8000 != 0 {
            let (three, one) = (k(self, 3), k(self, 1));
            let sel = self.s.m.bin(c, Bin::And, lane, three);
            let sh = self.s.m.bin(c, Bin::Shl, sel, one);
            let pattern = k(self, offset & 0xFF);
            let pick = self.s.m.bin(c, Bin::LShr, pattern, sh);
            let pick = self.s.m.bin(c, Bin::And, pick, three);
            let not3 = k(self, !3);
            let base = self.s.m.bin(c, Bin::And, lane, not3);
            self.s.m.bin(c, Bin::Or, base, pick)
        } else {
            let and = k(self, offset & 0x1F | !0x1F);
            let or = k(self, offset >> 5 & 0x1F);
            let xor = k(self, offset >> 10 & 0x1F);
            let t = self.s.m.bin(c, Bin::And, lane, and);
            let t = self.s.m.bin(c, Bin::Or, t, or);
            self.s.m.bin(c, Bin::Xor, t, xor)
        };
        Ok(if mesh_quad { self.quad_pick(x, src) } else { self.read_lane(x, src) })
    }

    /// The LDS dword at byte address `a + bytes`.
    // shortcut: LDS addresses are taken as dword-aligned and in bounds
    // (out-of-bounds console accesses read 0 and drop writes).
    fn lds_dword(&mut self, a: V, bytes: u32) -> V {
        let c = self.cur;
        let o = self.s.m.i32(bytes);
        let at = self.s.m.bin(c, Bin::Add, a, o);
        let two = self.s.m.i32(2);
        let index = self.s.m.bin(c, Bin::LShr, at, two);
        let lds = self.s.lds();
        self.s.m.element(c, lds, index)
    }

    /// `ds_read*`: `elements` from address `a` into VGPRs from `reg` on.
    fn lds_read(&mut self, inst: &Inst, a: V, elements: &[(u32, u32)], reg: u8, guard: Option<V>) {
        let mut n = 0;
        for &(off, w) in elements {
            for k in 0..w {
                let p = self.lds_dword(a, off + 4 * k);
                let x = self.s.m.load(self.cur, p);
                self.write(inst, Reg::V(reg + n), x, guard);
                n += 1;
            }
        }
    }

    /// `ds_write*`: `elements` of `data` to address `a`.
    fn lds_write(
        &mut self,
        inst: &Inst,
        a: V,
        elements: Vec<(u32, u32)>,
        data: &[Operand],
        guard: Option<V>,
    ) -> Result<(), Why> {
        let mut stores = Vec::new();
        for (e, (off, w)) in elements.into_iter().enumerate() {
            let vals = self.vgprs(inst, data.get(e).ok_or("an LDS write")?, w as u8)?;
            for (k, x) in vals.into_iter().enumerate() {
                stores.push((off + 4 * k as u32, x));
            }
        }
        let at = self.begin_guarded(guard);
        for (off, x) in stores {
            let p = self.lds_dword(a, off);
            self.s.m.store(self.cur, p, x);
        }
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// LDS atomics at address `a + offset`: the old value returns with
    /// `_rtn`.
    fn lds_atomic(
        &mut self,
        inst: &Inst,
        a: V,
        offset: u32,
        data: &[Operand],
        dst_reg: Option<u8>,
        guard: Option<V>,
    ) -> Result<(), Why> {
        let op = inst
            .name()
            .strip_prefix("ds_")
            .map(|n| n.replace("_rtn", ""))
            .ok_or("an LDS operation")?;
        let rmw = rmw(&op)?;
        let v0 = self.vgprs(inst, data.first().ok_or("an LDS atomic")?, 1)?[0];
        let v1 = match data.get(1) {
            Some(d) => Some(self.vgprs(inst, d, 1)?[0]),
            None => None,
        };
        let old_reg = dst_reg.and_then(|r| inst.read(Reg::V(r)).map(|v| (r, v)));
        let outside = match old_reg {
            Some((_, v)) => self.i32_of(v).unwrap_or(self.s.m.i32(0)),
            None => self.s.m.i32(0),
        };
        let p = self.lds_dword(a, offset);
        let at = self.begin_guarded(guard);
        let r = match rmw {
            Some(op) => self.s.m.atomic(self.cur, op, p, v0),
            // Compare with data0, store data1.
            None => self.s.m.cmpxchg(self.cur, p, v0, v1.ok_or("an LDS compare")?),
        };
        let r = self.end_guarded(at, &[(r, outside)])[0];
        if let Some(reg) = dst_reg {
            self.write(inst, Reg::V(reg), r, None);
        }
        Ok(())
    }
}

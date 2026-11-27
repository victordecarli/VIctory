//! The 32-bit vector ALU: the dispatch of each VALU instruction, DPP, float
//! arithmetic and conversions, lane reads and writes, and 64-bit sources.

use super::*;
use crate::gfx10::Valu;

/// One 32-bit vector ALU instruction as `valu_op` lowers it.
struct Op<'i> {
    inst: &'i Inst,
    v: &'i Valu,
    name: &'static str,
    /// The block the instruction is lowered in.
    c: usize,
    /// The lanes that write their result (none: all).
    guard: Option<V>,
}

impl Lower<'_> {
    /// `f` as an f16 in the low bits, rounded toward zero: the nearest-even
    /// conversion, one step back toward zero where it rounded away (which
    /// also turns an overflow to infinity into the largest finite half).
    fn f16_toward_zero(&mut self, f: V) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let h = self.s.op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
        let back = self.s.op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h]);
        let (fa, ba) = (self.unary(6, f), self.unary(6, back));
        // Ordered: never for NaN.
        let away = self.s.m.cmp(c, pred::OGT, ba, fa);
        let one = self.s.m.i32(1);
        let stepped = self.s.m.bin(c, Bin::Sub, h, one);
        self.s.m.select(c, away, stepped, h)
    }

    /// `0 × anything = 0`, as in D3D9.
    fn legacy_mul(&mut self, a: V, b: V) -> V {
        let c = self.cur;
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let zero = self.s.m.f32(0);
        let az = self.s.m.cmp(c, pred::OEQ, a, zero);
        let bz = self.s.m.cmp(c, pred::OEQ, b, zero);
        let z = self.s.m.bin(c, Bin::Or, az, bz);
        self.s.m.select(c, z, zero, m)
    }

    pub(super) fn valu(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            return Err("a vector ALU instruction without VALU fields");
        };
        let name = inst.name();
        if self.x.res.waterfall.compares.contains(&(b, i)) {
            // The first lane's dwords of a waterfall loop are each lane's
            // own (see `resource::Waterfall`).
            self.lane(inst, Some(true));
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("v_cmp_").or(name.strip_prefix("v_cmpx_")) {
            let r =
                if v.dpp.is_some() { Err("a compare with DPP") } else { self.compare(inst, rest) };
            if let Err(why) = r {
                for &(_, w) in &inst.writes {
                    self.lane_fail.insert(w, why);
                }
            }
            return r;
        }
        let mut guard = self.guard(b, i)?;
        if let Some(d) = v.dpp {
            let (x, enable) = self.dpp(inst, d)?;
            guard = match (guard, enable) {
                (Some(g), Some(e)) => Some(self.s.m.bin(self.cur, Bin::And, g, e)),
                (g, e) => g.or(e),
            };
            self.src0 = Some(x);
        }
        let r = self.valu_op(inst, guard);
        self.src0 = None;
        r
    }

    /// DPP: src0 as read from another lane, and which lanes write their
    /// result (none: all).
    fn dpp(&mut self, inst: &Inst, dpp: Dpp) -> Result<(V, Option<V>), Why> {
        let Fields::Valu(v) = &inst.fields else {
            return Err("a vector ALU instruction without VALU fields");
        };
        let Some(&Operand::Vector { reg, .. }) = v.src.first() else {
            return Err("a DPP operand");
        };
        let x = self.i32_of(inst.read(Reg::V(reg)).ok_or("a DPP operand")?)?;
        // Quad permutations and DPP8 stay in groups the pixel layout shares.
        let quad = matches!(dpp, Dpp::Row { ctrl: 0..=0xFF, .. });
        let mesh_quad = self.s.mesh.is_some() && quad;
        if !mesh_quad {
            self.cross_lane(quad)?;
        }
        let c = self.cur;
        let lane = self.lane_index();
        let (sixteen_mask, low) = (self.s.m.i32(!15), self.s.m.i32(15));
        let row = self.s.m.bin(c, Bin::And, lane, sixteen_mask);
        let pos = self.s.m.bin(c, Bin::And, lane, low);
        let at = (lane, row, pos, low);
        // Source lane, and whether it lies in the row.
        let (src, inside, masks, zero) = match dpp {
            Dpp::Lanes { sel, .. } => {
                let (seven, not7, three) = (self.s.m.i32(7), self.s.m.i32(!7), self.s.m.i32(3));
                let p = self.s.m.bin(c, Bin::And, lane, seven);
                let sh = self.s.m.bin(c, Bin::Mul, p, three);
                let sel = self.s.m.i32(sel);
                let pick = self.s.m.bin(c, Bin::LShr, sel, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, seven);
                let base = self.s.m.bin(c, Bin::And, lane, not7);
                (self.s.m.bin(c, Bin::Or, base, pick), None, None, false)
            }
            Dpp::Row { ctrl, row_mask, bank_mask, bound_ctrl, .. } => {
                let (src, inside) = self.dpp_row_source(c, at, ctrl)?;
                let masks = (row_mask != 0xF || bank_mask != 0xF).then_some((row_mask, bank_mask));
                (src, inside, masks, bound_ctrl)
            }
        };
        let mut r = if mesh_quad { self.quad_pick(x, src) } else { self.read_lane(x, src) };
        let mut enable = None;
        if let Some(inside) = inside {
            if zero {
                let z = self.s.m.i32(0);
                r = self.s.m.select(c, inside, r, z);
            } else {
                enable = Some(inside);
            }
        }
        if let Some((rows, banks)) = masks {
            let on = self.dpp_write_mask(c, lane, rows, banks);
            enable = Some(match enable {
                Some(e) => self.s.m.bin(c, Bin::And, e, on),
                None => on,
            });
        }
        Ok((r, enable))
    }

    /// The source lane of a DPP row pattern `ctrl`, and for the row shifts
    /// whether it lies in the row. `at` is the lane, its row's first lane,
    /// its position in the row and the constant 15.
    fn dpp_row_source(
        &mut self,
        c: usize,
        at: (V, V, V, V),
        ctrl: u16,
    ) -> Result<(V, Option<V>), Why> {
        let (lane, row, pos, low) = at;
        let n = u32::from(ctrl & 15);
        Ok(match ctrl {
            0..=0xFF => {
                let (three, one, not3) = (self.s.m.i32(3), self.s.m.i32(1), self.s.m.i32(!3));
                let p = self.s.m.bin(c, Bin::And, lane, three);
                let sh = self.s.m.bin(c, Bin::Shl, p, one);
                let perm = self.s.m.i32(u32::from(ctrl));
                let pick = self.s.m.bin(c, Bin::LShr, perm, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, three);
                let base = self.s.m.bin(c, Bin::And, lane, not3);
                (self.s.m.bin(c, Bin::Or, base, pick), None)
            }
            0x101..=0x10F | 0x111..=0x11F => {
                let nn = self.s.m.i32(n);
                let left = ctrl < 0x110;
                let (op, p) = if left { (Bin::Add, pred::IULT) } else { (Bin::Sub, pred::IUGE) };
                let s = self.s.m.bin(c, op, lane, nn);
                // Left: pos + n < 16; right: pos >= n.
                let bound = if left { self.s.m.i32(16 - n) } else { nn };
                (s, Some(self.s.m.cmp(c, p, pos, bound)))
            }
            0x121..=0x12F => {
                let nn = self.s.m.i32(n);
                let t = self.s.m.bin(c, Bin::Sub, pos, nn);
                let t = self.s.m.bin(c, Bin::And, t, low);
                (self.s.m.bin(c, Bin::Or, row, t), None)
            }
            0x140 | 0x141 => {
                let m = self.s.m.i32(if ctrl == 0x140 { 15 } else { 7 });
                let t = self.s.m.bin(c, Bin::Xor, lane, m);
                (t, None)
            }
            0x150..=0x15F => {
                let nn = self.s.m.i32(n);
                (self.s.m.bin(c, Bin::Or, row, nn), None)
            }
            0x160..=0x16F => {
                let nn = self.s.m.i32(n);
                (self.s.m.bin(c, Bin::Xor, lane, nn), None)
            }
            _ => return Err("a DPP lane pattern"),
        })
    }

    /// Whether `lane` writes under DPP row (16 lanes) and bank (4 lanes of
    /// a row) write masks.
    fn dpp_write_mask(&mut self, c: usize, lane: V, rows: u8, banks: u8) -> V {
        let (four, two, three) = (self.s.m.i32(4), self.s.m.i32(2), self.s.m.i32(3));
        let ri = self.s.m.bin(c, Bin::LShr, lane, four);
        let ri = self.s.m.bin(c, Bin::And, ri, three);
        let bi = self.s.m.bin(c, Bin::LShr, lane, two);
        let bi = self.s.m.bin(c, Bin::And, bi, three);
        let (rm, bm) = (self.s.m.i32(u32::from(rows)), self.s.m.i32(u32::from(banks)));
        let rb = self.s.m.bin(c, Bin::LShr, rm, ri);
        let bb = self.s.m.bin(c, Bin::LShr, bm, bi);
        let both = self.s.m.bin(c, Bin::And, rb, bb);
        let (one, z) = (self.s.m.i32(1), self.s.m.i32(0));
        let both = self.s.m.bin(c, Bin::And, both, one);
        self.s.m.cmp(c, pred::NE, both, z)
    }

    fn valu_op(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            return Err("a vector ALU instruction without VALU fields");
        };
        let name = inst.name();
        let o = Op { inst, v, name, c: self.cur, guard };
        if let Some(r) = self.valu16(inst, name, guard) {
            return r;
        }
        if let Some(op) = name.strip_prefix("v_pk_") {
            return self.packed16(inst, op, guard);
        }
        for family in [Self::valu_move, Self::valu_float, Self::valu_convert, Self::valu_lane] {
            if let Some(r) = family(self, &o) {
                return r;
            }
        }
        match name {
            "v_ffbh_u32" | "v_ffbh_i32" => {
                // Counted from the top bit, -1 when none (DXIL's too).
                let a = self.src(inst, 0)?;
                let op = if name == "v_ffbh_u32" { 33 } else { 34 };
                let r = self.s.op(o.c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                self.write_i(inst, r, guard)
            }
            "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" | "v_add_co_ci_u32"
            | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32" => self.add_co(&o),
            _ => {
                let r = self.integer(inst, name)?;
                self.write_i(inst, r, guard)
            }
        }
    }

    /// Moves, selects and 64-bit shifts; `None` for another family.
    fn valu_move(&mut self, o: &Op) -> Option<Result<(), Why>> {
        let (inst, v) = (o.inst, o.v);
        Some(match o.name {
            "v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64" => self.shift64(o),
            "v_movrels_b32" | "v_movreld_b32" | "v_movrelsd_b32" => self.movrel(o),
            "v_swap_b32" => self.swap(o),
            "v_mov_b32" => {
                if v.abs != 0 || v.neg != 0 || v.clamp || v.omod != 0 {
                    self.fsrc(inst, 0).and_then(|a| self.write_f(inst, a, o.guard))
                } else {
                    self.src(inst, 0).and_then(|a| self.write_i(inst, a, o.guard))
                }
            }
            "v_cndmask_b32" => self.cndmask(o),
            _ => return None,
        })
    }

    /// `v_lshlrev_b64`, `v_lshrrev_b64` and `v_ashrrev_i64`.
    fn shift64(&mut self, o: &Op) -> Result<(), Why> {
        let (inst, c) = (o.inst, o.c);
        let n = self.src(inst, 0)?;
        let x = self.src64(inst, 1)?;
        let i64t = self.s.m.int(64);
        let k63 = self.s.m.i32(63);
        let n = self.s.m.bin(c, Bin::And, n, k63);
        let n = self.s.m.cast(c, Cast::ZExt, n, i64t);
        let op = match o.name {
            "v_lshlrev_b64" => Bin::Shl,
            "v_lshrrev_b64" => Bin::LShr,
            _ => Bin::AShr,
        };
        let r = self.s.m.bin(c, op, x, n);
        let Some(Operand::Vector { reg, .. }) = o.v.dst else {
            return Err("a 64-bit result");
        };
        let (lo, hi) = self.split64(r);
        self.write(inst, Reg::V(reg), lo, o.guard);
        self.write(inst, Reg::V(reg + 1), hi, o.guard);
        Ok(())
    }

    /// `v_movrels_b32`, `v_movreld_b32` and `v_movrelsd_b32`: M0 indexes
    /// the source (movrels), the destination (movreld) or both, from the
    /// encoded register up to the program's VGPR count (the registers the
    /// instruction reads or writes).
    fn movrel(&mut self, o: &Op) -> Result<(), Why> {
        let (inst, c) = (o.inst, o.c);
        let m0 = self.i32_of(inst.read(Reg::S(crate::ir::M0)).ok_or("M0")?)?;
        let (Some(&Operand::Vector { reg: sreg, .. }), Some(Operand::Vector { reg: dreg, .. })) =
            (o.v.src.first(), o.v.dst)
        else {
            return Err("an M0-relative move");
        };
        let indexed_src = o.name != "v_movreld_b32";
        let value = if indexed_src {
            let mut r = self.s.m.i32(0);
            for k in 0..=255u8.saturating_sub(sreg) {
                let Some(x) = inst.read(Reg::V(sreg + k)) else {
                    break;
                };
                let x = self.i32_of(x)?;
                let kk = self.s.m.i32(u32::from(k));
                let is = self.s.m.cmp(c, pred::EQ, m0, kk);
                r = self.s.m.select(c, is, x, r);
            }
            r
        } else {
            self.src(inst, 0)?
        };
        if o.name == "v_movrels_b32" {
            return self.write_i(inst, value, o.guard);
        }
        for k in 0..=255u8.saturating_sub(dreg) {
            let Some(_) = inst.write(Reg::V(dreg + k)) else {
                break;
            };
            let old = match inst.read(Reg::V(dreg + k)) {
                Some(o) => self.i32_of(o)?,
                None => self.s.m.i32(0),
            };
            let kk = self.s.m.i32(u32::from(k));
            let is = self.s.m.cmp(c, pred::EQ, m0, kk);
            let x = self.s.m.select(c, is, value, old);
            self.write_placed(inst, Reg::V(dreg + k), x, o.guard);
        }
        Ok(())
    }

    /// `v_swap_b32`.
    fn swap(&mut self, o: &Op) -> Result<(), Why> {
        let inst = o.inst;
        let (Some(&Operand::Vector { reg: sreg, .. }), Some(Operand::Vector { reg: dreg, .. })) =
            (o.v.src.first(), o.v.dst)
        else {
            return Err("v_swap_b32");
        };
        let a = self.i32_of(inst.read(Reg::V(dreg)).ok_or("v_swap_b32")?)?;
        let b = self.i32_of(inst.read(Reg::V(sreg)).ok_or("v_swap_b32")?)?;
        self.write_placed(inst, Reg::V(dreg), b, o.guard);
        if dreg != sreg {
            self.write_placed(inst, Reg::V(sreg), a, o.guard);
        }
        Ok(())
    }

    /// `v_cndmask_b32`: src1 where the lane mask is set, else src0.
    fn cndmask(&mut self, o: &Op) -> Result<(), Why> {
        let (inst, v, c) = (o.inst, o.v, o.c);
        let mask = match v.src.get(2) {
            Some(&Operand::Scalar { code, .. }) => inst.read(Reg::S(code)),
            _ => None,
        };
        let m = mask
            .and_then(|m| self.x.st.masks.get(&m).copied())
            .ok_or("a select on an untracked mask")?;
        let cond = self.boolean(m, c)?;
        if v.abs != 0 || v.neg != 0 {
            let (a, bb) = (self.fsrc(inst, 0)?, self.fsrc(inst, 1)?);
            let r = self.s.m.select(c, cond, bb, a);
            return self.write_f(inst, r, o.guard);
        }
        let (a, bb) = (self.src(inst, 0)?, self.src(inst, 1)?);
        let r = self.s.m.select(c, cond, bb, a);
        self.write_i(inst, r, o.guard)
    }

    /// The add and subtract with a carry (borrow) out, which is a lane
    /// mask: this lane's bit.
    fn add_co(&mut self, o: &Op) -> Result<(), Why> {
        let (inst, v, c, name) = (o.inst, o.v, o.c, o.name);
        let (mut a, mut bb) = (self.src(inst, 0)?, self.src(inst, 1)?);
        if name.contains("subrev") {
            std::mem::swap(&mut a, &mut bb);
        }
        let cin = if name.ends_with("_ci_u32") {
            let m = match v.src.get(2) {
                Some(&Operand::Scalar { code, .. }) => inst.read(Reg::S(code)),
                _ => None,
            };
            let m = m
                .and_then(|m| self.x.st.masks.get(&m).copied())
                .ok_or("a carry-in from an untracked mask")?;
            Some(self.boolean(m, c)?)
        } else {
            None
        };
        let (r, carry) = self.add_sub_carry(c, name.starts_with("v_add"), a, bb, cin);
        self.write_i(inst, r, o.guard)?;
        if let Some(Operand::Scalar { code, .. }) = v.carry
            && let Some(w) = inst.write(Reg::S(code))
        {
            self.lane_bits.insert(w, carry);
        }
        for &(_, w) in &inst.writes {
            self.vals.entry(w).or_insert(Val::Missing("a lane mask read as data"));
        }
        Ok(())
    }

    /// Float arithmetic; `None` for another family.
    fn valu_float(&mut self, o: &Op) -> Option<Result<(), Why>> {
        Some(match o.name {
            "v_add_f32" | "v_sub_f32" | "v_subrev_f32" | "v_mul_f32" | "v_mul_legacy_f32"
            | "v_max_f32" | "v_min_f32" => self.float2(o),
            "v_fma_f32" | "v_fmaak_f32" | "v_fmamk_f32" | "v_mad_f32" | "v_madak_f32"
            | "v_madmk_f32" | "v_mad_legacy_f32" | "v_max3_f32" | "v_min3_f32" | "v_med3_f32" => {
                self.float3(o)
            }
            "v_fmac_f32" | "v_mac_f32" | "v_mac_legacy_f32" => self.mac(o),
            "v_rcp_f32" | "v_rcp_iflag_f32" | "v_rsq_f32" | "v_sqrt_f32" | "v_log_f32"
            | "v_exp_f32" | "v_fract_f32" | "v_trunc_f32" | "v_floor_f32" | "v_ceil_f32"
            | "v_rndne_f32" | "v_sin_f32" | "v_cos_f32" => self.float1(o),
            "v_cubeid_f32" | "v_cubesc_f32" | "v_cubetc_f32" | "v_cubema_f32" => self.cube(o),
            _ => return None,
        })
    }

    /// Float operations of two sources.
    fn float2(&mut self, o: &Op) -> Result<(), Why> {
        let (inst, c) = (o.inst, o.c);
        let (a, bb) = (self.fsrc(inst, 0)?, self.fsrc(inst, 1)?);
        let r = match o.name {
            "v_add_f32" => self.s.m.bin(c, Bin::Add, a, bb),
            "v_sub_f32" => self.s.m.bin(c, Bin::Sub, a, bb),
            "v_subrev_f32" => self.s.m.bin(c, Bin::Sub, bb, a),
            "v_mul_f32" => self.s.m.bin(c, Bin::Mul, a, bb),
            "v_mul_legacy_f32" => self.legacy_mul(a, bb),
            "v_max_f32" => self.max_min(true, a, bb),
            _ => self.max_min(false, a, bb),
        };
        self.write_f(inst, r, o.guard)
    }

    /// Float operations of three sources: multiply-adds and the
    /// three-way max, min and median.
    fn float3(&mut self, o: &Op) -> Result<(), Why> {
        let (inst, c, name) = (o.inst, o.c, o.name);
        let (a, bb, cc) = (self.fsrc(inst, 0)?, self.fsrc(inst, 1)?, self.fsrc(inst, 2)?);
        let r = match name {
            // shortcut: DXIL has no fused f32 multiply-add; FMad lets the driver fuse.
            "v_fma_f32" | "v_fmaak_f32" | "v_fmamk_f32" => self.fmad(a, bb, cc),
            "v_mad_f32" | "v_madak_f32" | "v_madmk_f32" | "v_mad_legacy_f32" => {
                let m = if name == "v_mad_legacy_f32" {
                    self.legacy_mul(a, bb)
                } else {
                    self.s.m.bin(c, Bin::Mul, a, bb)
                };
                self.s.m.bin(c, Bin::Add, m, cc)
            }
            "v_max3_f32" => {
                let t = self.max_min(true, a, bb);
                self.max_min(true, t, cc)
            }
            "v_min3_f32" => {
                let t = self.max_min(false, a, bb);
                self.max_min(false, t, cc)
            }
            _ => self.med3(a, bb, cc),
        };
        self.write_f(inst, r, o.guard)
    }

    /// `v_fmac_f32`, `v_mac_f32` and `v_mac_legacy_f32`: the destination
    /// is the accumulator.
    fn mac(&mut self, o: &Op) -> Result<(), Why> {
        let (inst, c) = (o.inst, o.c);
        let (a, bb) = (self.fsrc(inst, 0)?, self.fsrc(inst, 1)?);
        let Some(Operand::Vector { reg, .. }) = o.v.dst else {
            return Err("an accumulator");
        };
        let acc = inst.read(Reg::V(reg)).ok_or("an accumulator")?;
        let acc = match self.floats.get(&acc) {
            Some(&x) => x,
            None => {
                let x = self.i32_of(acc)?;
                self.as_float(x)
            }
        };
        let r = match o.name {
            "v_fmac_f32" => self.fmad(a, bb, acc),
            "v_mac_f32" => {
                let m = self.s.m.bin(c, Bin::Mul, a, bb);
                self.s.m.bin(c, Bin::Add, m, acc)
            }
            _ => {
                let m = self.legacy_mul(a, bb);
                self.s.m.bin(c, Bin::Add, m, acc)
            }
        };
        self.write_f(inst, r, o.guard)
    }

    /// Float operations of one source: reciprocal, roots, logarithm and
    /// exponent, rounding, fraction and trigonometry.
    fn float1(&mut self, o: &Op) -> Result<(), Why> {
        let a = self.fsrc(o.inst, 0)?;
        let op = match o.name {
            "v_rcp_f32" | "v_rcp_iflag_f32" => {
                // shortcut: the hardware reciprocal is approximate (1 ULP); this divides.
                let one = self.s.m.f32(1.0f32.to_bits());
                let r = self.s.m.bin(o.c, Bin::SDiv, one, a);
                return self.write_f(o.inst, r, o.guard);
            }
            "v_rsq_f32" => 25,
            "v_sqrt_f32" => 24,
            "v_log_f32" => 23,
            "v_exp_f32" => 21,
            "v_fract_f32" => 22,
            "v_trunc_f32" => 29,
            "v_floor_f32" => 27,
            "v_ceil_f32" => 28,
            "v_rndne_f32" => 26,
            "v_sin_f32" => 13,
            _ => 12,
        };
        let r = match op {
            12 | 13 => self.trig_turns(a, op == 12),
            22 => self.fract(a, 1.0, f32::from_bits(0x3F7F_FFFF)),
            _ => self.unary(op, a),
        };
        self.write_f(o.inst, r, o.guard)
    }

    /// Cube-map face coordinates (AMD's ISA): the major axis is z when
    /// |z| >= |x| and |z| >= |y|, else y when |y| >= |x|, else x.
    fn cube(&mut self, o: &Op) -> Result<(), Why> {
        let (inst, c) = (o.inst, o.c);
        let (x, y, z) = (self.fsrc(inst, 0)?, self.fsrc(inst, 1)?, self.fsrc(inst, 2)?);
        let (ax, ay, az) = (self.unary(6, x), self.unary(6, y), self.unary(6, z));
        let zx = self.s.m.cmp(c, pred::OGE, az, ax);
        let zy = self.s.m.cmp(c, pred::OGE, az, ay);
        let zmaj = self.s.m.bin(c, Bin::And, zx, zy);
        let ymaj = self.s.m.cmp(c, pred::OGE, ay, ax);
        let zero = self.s.m.f32(0);
        let neg = |l: &mut Self, v: V| l.s.m.cmp(c, pred::OLT, v, zero);
        let minus = |l: &mut Self, v: V| {
            let z = l.s.m.f32(0x8000_0000);
            l.s.m.bin(c, Bin::Sub, z, v)
        };
        let pick = |l: &mut Self, zv: V, yv: V, xv: V| {
            let t = l.s.m.select(c, ymaj, yv, xv);
            l.s.m.select(c, zmaj, zv, t)
        };
        let r = match o.name {
            "v_cubeid_f32" => {
                let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
                let (nz, ny, nx) = (neg(self, z), neg(self, y), neg(self, x));
                let (f5, f4, f3, f2, f1, f0) = (
                    k(self, 5.0),
                    k(self, 4.0),
                    k(self, 3.0),
                    k(self, 2.0),
                    k(self, 1.0),
                    k(self, 0.0),
                );
                let zf = self.s.m.select(c, nz, f5, f4);
                let yf = self.s.m.select(c, ny, f3, f2);
                let xf = self.s.m.select(c, nx, f1, f0);
                pick(self, zf, yf, xf)
            }
            "v_cubesc_f32" => {
                let (nz, nx) = (neg(self, z), neg(self, x));
                let (mx, mz) = (minus(self, x), minus(self, z));
                let zf = self.s.m.select(c, nz, mx, x);
                let xf = self.s.m.select(c, nx, z, mz);
                pick(self, zf, x, xf)
            }
            "v_cubetc_f32" => {
                let ny = neg(self, y);
                let (my, mz) = (minus(self, y), minus(self, z));
                let yf = self.s.m.select(c, ny, mz, z);
                pick(self, my, yf, my)
            }
            _ => {
                let t = pick(self, z, y, x);
                let two = self.s.m.f32(2.0f32.to_bits());
                self.s.m.bin(c, Bin::Mul, t, two)
            }
        };
        self.write_f(inst, r, o.guard)
    }

    /// Conversions between floats, integers and halves; `None` for another
    /// family (and for the f16 conversions with op_sel, which `integer`
    /// reports).
    fn valu_convert(&mut self, o: &Op) -> Option<Result<(), Why>> {
        let v = o.v;
        Some(match o.name {
            "v_cvt_i32_f32" | "v_cvt_u32_f32" | "v_cvt_flr_i32_f32" | "v_cvt_rpi_i32_f32" => {
                self.float_to_int(o)
            }
            "v_cvt_f32_i32" | "v_cvt_f32_u32" | "v_cvt_f32_ubyte0" | "v_cvt_f32_ubyte1"
            | "v_cvt_f32_ubyte2" | "v_cvt_f32_ubyte3" => self.int_to_float(o),
            "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => self.fma_mix(o),
            "v_cvt_pkrtz_f16_f32" => self.pkrtz(o),
            "v_cvt_f16_f32" if v.opsel == 0 => self.f32_to_f16(o),
            "v_pack_b32_f16" => self.pack_f16(o),
            "v_cvt_f32_f16" if v.opsel == 0 => self.src(o.inst, 0).and_then(|a| {
                let f32t = self.s.t.f32;
                let r = self.s.op(o.c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[a]);
                self.write_f(o.inst, r, o.guard)
            }),
            _ => return None,
        })
    }

    /// Float to 32-bit integer, truncated, floored or rounded.
    fn float_to_int(&mut self, o: &Op) -> Result<(), Why> {
        let (c, name) = (o.c, o.name);
        let mut a = self.fsrc(o.inst, 0)?;
        if name == "v_cvt_rpi_i32_f32" {
            let h = self.s.m.f32(0.5f32.to_bits());
            a = self.s.m.bin(c, Bin::Add, a, h);
        }
        if name != "v_cvt_i32_f32" && name != "v_cvt_u32_f32" {
            a = self.unary(27, a);
        }
        // D3D's conversions saturate and give 0 for NaN, as the
        // hardware does (checked on the RX 6700 XT and WARP).
        let op = if name == "v_cvt_u32_f32" { Cast::FpToUi } else { Cast::FpToSi };
        let r = self.s.m.cast(c, op, a, self.s.t.i32);
        self.write_i(o.inst, r, o.guard)
    }

    /// 32-bit integer, or one byte of it, to float.
    fn int_to_float(&mut self, o: &Op) -> Result<(), Why> {
        let (c, name) = (o.c, o.name);
        let mut a = self.src(o.inst, 0)?;
        if let Some(n) = name.strip_prefix("v_cvt_f32_ubyte") {
            let sh = self.s.m.i32(8 * u32::from(n.as_bytes()[0] - b'0'));
            let ff = self.s.m.i32(0xff);
            a = self.s.m.bin(c, Bin::LShr, a, sh);
            a = self.s.m.bin(c, Bin::And, a, ff);
        }
        let op = if name == "v_cvt_f32_i32" { Cast::SiToFp } else { Cast::UiToFp };
        let r = self.s.m.cast(c, op, a, self.s.t.f32);
        self.write_f(o.inst, r, o.guard)
    }

    /// `v_fma_mix_f32`, `v_fma_mixlo_f16` and `v_fma_mixhi_f16`: each
    /// source is f32, or with op_sel_hi the low (op_sel: high) f16 half;
    /// neg_hi is the absolute value here.
    fn fma_mix(&mut self, o: &Op) -> Result<(), Why> {
        let (inst, v, c) = (o.inst, o.v, o.c);
        let mut xs = Vec::new();
        for i in 0..3 {
            let op = v.src.get(i).ok_or("a missing operand")?;
            let x = self.operand(inst, op)?;
            let mut x = if v.opsel_hi >> i & 1 != 0 {
                let h = if v.opsel >> i & 1 != 0 {
                    let k16 = self.s.m.i32(16);
                    self.s.m.bin(c, Bin::LShr, x, k16)
                } else {
                    x
                };
                self.s.op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h])
            } else {
                self.as_float(x)
            };
            if v.neg_hi >> i & 1 != 0 {
                x = self.unary(6, x);
            }
            if v.neg >> i & 1 != 0 {
                let z = self.s.m.f32(0x8000_0000);
                x = self.s.m.bin(c, Bin::Sub, z, x);
            }
            xs.push(x);
        }
        let mut r = self.fmad(xs[0], xs[1], xs[2]);
        if o.name == "v_fma_mix_f32" {
            return self.write_f(inst, r, o.guard);
        }
        if v.clamp {
            r = self.unary(7, r);
        }
        // The f16 result goes to one half; the other half stays.
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a mixed-precision result");
        };
        let old = match inst.read(Reg::V(reg)) {
            Some(o) => self.i32_of(o)?,
            None => self.s.m.i32(0),
        };
        let h = self.s.op(c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[r]);
        let (keep, h) = if o.name == "v_fma_mixhi_f16" {
            let k16 = self.s.m.i32(16);
            (0xFFFF, self.s.m.bin(c, Bin::Shl, h, k16))
        } else {
            (0xFFFF_0000, h)
        };
        let keep = self.s.m.i32(keep);
        let old = self.s.m.bin(c, Bin::And, old, keep);
        let x = self.s.m.bin(c, Bin::Or, old, h);
        self.write(inst, Reg::V(reg), x, o.guard);
        Ok(())
    }

    /// `v_cvt_pkrtz_f16_f32`: two floats to halves rounded toward zero.
    fn pkrtz(&mut self, o: &Op) -> Result<(), Why> {
        let c = o.c;
        let (a, bb) = (self.fsrc(o.inst, 0)?, self.fsrc(o.inst, 1)?);
        let lo = self.f16_toward_zero(a);
        let hi = self.f16_toward_zero(bb);
        let sh = self.s.m.i32(16);
        let hi = self.s.m.bin(c, Bin::Shl, hi, sh);
        let r = self.s.m.bin(c, Bin::Or, lo, hi);
        self.write_i(o.inst, r, o.guard)
    }

    /// `v_cvt_f16_f32` without op_sel: rounded to nearest even (the MODE
    /// default) into the low half.
    fn f32_to_f16(&mut self, o: &Op) -> Result<(), Why> {
        let a = self.fsrc(o.inst, 0)?;
        let r = self.s.op(o.c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[a]);
        let Some(Operand::Vector { reg, .. }) = o.v.dst else {
            return Err("a 16-bit result outside a VGPR");
        };
        let r = self.low_half(o.inst, reg, r);
        self.write_i(o.inst, r, o.guard)
    }

    /// `v_pack_b32_f16`: two f16 sources (op_sel: their high halves; abs
    /// and neg on the sign bit) into the low and high half.
    fn pack_f16(&mut self, o: &Op) -> Result<(), Why> {
        let (inst, v, c) = (o.inst, o.v, o.c);
        let mut halves = Vec::new();
        for k in 0..2 {
            let op = v.src.get(k).ok_or("a missing operand")?;
            let mut x = self.operand(inst, op)?;
            if v.opsel >> k & 1 != 0 {
                let k16 = self.s.m.i32(16);
                x = self.s.m.bin(c, Bin::LShr, x, k16);
            }
            let mask = if v.abs >> k & 1 != 0 { 0x7FFF } else { 0xFFFF };
            let mask = self.s.m.i32(mask);
            x = self.s.m.bin(c, Bin::And, x, mask);
            if v.neg >> k & 1 != 0 {
                let sign = self.s.m.i32(0x8000);
                x = self.s.m.bin(c, Bin::Xor, x, sign);
            }
            halves.push(x);
        }
        let k16 = self.s.m.i32(16);
        let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
        let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
        self.write_i(inst, r, o.guard)
    }

    /// Lane reads and writes; `None` for another family.
    fn valu_lane(&mut self, o: &Op) -> Option<Result<(), Why>> {
        let inst = o.inst;
        Some(match o.name {
            "v_readlane_b32" => self.readlane(o),
            "v_writelane_b32" => self.writelane(o),
            "v_readfirstlane_b32"
                if inst
                    .writes
                    .first()
                    .is_some_and(|w| self.x.res.waterfall.reads.contains(&w.1)) =>
            {
                // A waterfall loop's read of a descriptor dword: the lane's own.
                self.src(inst, 0).and_then(|a| self.write_i(inst, a, o.guard))
            }
            "v_readfirstlane_b32" => self.readfirstlane(o),
            _ => return None,
        })
    }

    /// `v_readlane_b32`.
    fn readlane(&mut self, o: &Op) -> Result<(), Why> {
        let (a, lane) = (self.src(o.inst, 0)?, self.src(o.inst, 1)?);
        // A computed lane in a pixel or vertex program is a host lane
        // (from a mask or `v_mbcnt`); a constant one is the console's.
        if self.mask_data == MaskData::Host && !matches!(lane, V::Const(_)) {
            self.s.flags |= flags::WAVE_OPS;
        } else {
            self.cross_lane(false)?;
        }
        let r = if self.halves.is_some() {
            self.console_read_lane(a, lane)
        } else {
            self.read_lane(a, lane)
        };
        self.write_i(o.inst, r, o.guard)
    }

    /// `v_writelane_b32`: the lane given takes the scalar; the others keep
    /// their value.
    fn writelane(&mut self, o: &Op) -> Result<(), Why> {
        let (inst, c) = (o.inst, o.c);
        let (a, lane) = (self.src(inst, 0)?, self.src(inst, 1)?);
        let Some(Operand::Vector { reg, .. }) = o.v.dst else {
            return Err("a lane write");
        };
        let old = match inst.read(Reg::V(reg)) {
            Some(o) => self.i32_of(o)?,
            None => self.s.m.i32(0),
        };
        self.cross_lane(false)?;
        let me = self.lane_index();
        let is = self.s.m.cmp(c, pred::EQ, me, lane);
        let r = self.s.m.select(c, is, a, old);
        self.write_i(inst, r, o.guard)
    }

    /// `v_readfirstlane_b32` outside waterfall loops.
    fn readfirstlane(&mut self, o: &Op) -> Result<(), Why> {
        let a = self.src(o.inst, 0)?;
        let r = if self.halves.is_some() {
            self.console_first_lane(a, o.guard)
        } else {
            self.s.flags |= flags::WAVE_OPS;
            self.s.op(o.c, "waveReadLaneFirst.i32", Attr::NoUnwind, self.s.t.i32, 118, &[a])
        };
        self.write_i(o.inst, r, o.guard)
    }

    /// A 64-bit source (a register pair; inline constants sign-extend).
    pub(super) fn src64(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            return Err("a vector ALU instruction without VALU fields");
        };
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 || v.sdwa.is_some() || v.dpp.is_some() {
            return Err("64-bit operand modifiers");
        }
        let c = self.cur;
        let i64t = self.s.m.int(64);
        self.s.flags |= flags::INT64;
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let pair = |l: &mut Self, r: Reg, r1: Reg| -> Result<V, Why> {
            let lo = l.i32_of(inst.read(r).ok_or("an unread operand")?)?;
            let hi = l.i32_of(inst.read(r1).ok_or("an unread operand")?)?;
            let lo = l.s.m.cast(c, Cast::ZExt, lo, i64t);
            let hi = l.s.m.cast(c, Cast::ZExt, hi, i64t);
            let k32 = l.s.m.int_const(i64t, 32);
            let hi = l.s.m.bin(c, Bin::Shl, hi, k32);
            Ok(l.s.m.bin(c, Bin::Or, lo, hi))
        };
        match op {
            Operand::Vector { reg, .. } => pair(self, Reg::V(reg), Reg::V(reg + 1)),
            Operand::Scalar { code, .. } if code != NULL => {
                pair(self, Reg::S(code), Reg::S(code + 1))
            }
            Operand::Const(k @ 128..=208) => {
                let x = crate::resource::inline(k).ok_or("a special operand")?;
                Ok(self.s.m.int_const(i64t, i64::from(x as i32)))
            }
            _ => {
                // Literals and the null register: zero-extended.
                let x = self.operand(inst, &op)?;
                Ok(self.s.m.cast(c, Cast::ZExt, x, i64t))
            }
        }
    }

    /// The low and high dwords of an i64.
    fn split64(&mut self, x: V) -> (V, V) {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let k32 = self.s.m.int_const(i64t, 32);
        let lo = self.s.m.cast(c, Cast::Trunc, x, self.s.t.i32);
        let hi = self.s.m.bin(c, Bin::LShr, x, k32);
        let hi = self.s.m.cast(c, Cast::Trunc, hi, self.s.t.i32);
        (lo, hi)
    }

    pub(super) fn lane(&mut self, inst: &Inst, value: Option<bool>) {
        let b = self.s.m.bool(value == Some(true));
        self.lane_bits_for(inst, b);
    }

    pub(super) fn lane_bits_for(&mut self, inst: &Inst, r: V) {
        let Fields::Valu(v) = &inst.fields else {
            return;
        };
        let code = if inst.name().starts_with("v_cmpx") {
            Some(EXEC)
        } else {
            match v.dst {
                Some(Operand::Scalar { code, .. }) => Some(code),
                _ => None,
            }
        };
        if let Some(lo) = code.and_then(|c| inst.write(Reg::S(c))) {
            self.lane_bits.insert(lo, r);
        }
        for &(_, w) in &inst.writes {
            self.vals.entry(w).or_insert(Val::Missing("a lane mask read as data"));
        }
    }
}

//! The scalar ALU (SOP1, SOP2, SOPK and SOPC): 32- and 64-bit integer and
//! bit operations, scalar compares and the SCC they set.

use super::*;

/// One scalar ALU instruction: its fields and the values it writes.
struct Sop<'i> {
    inst: &'i Inst,
    name: &'static str,
    dst: &'i Option<Operand>,
    src: &'i [Operand],
    imm: Option<u16>,
    /// The block the instruction is lowered in.
    c: usize,
    scc: Option<Value>,
    /// The destination, and the high dword of a 64-bit one.
    lo: Option<Value>,
    hi: Option<Value>,
}

/// The binary operation of a bitwise `s_<op>_b32`/`_b64` by its name.
fn logic_op(name: &str) -> Bin {
    match &name[2..5] {
        "and" | "nan" => Bin::And,
        "or_" | "orn" | "nor" => Bin::Or,
        _ => Bin::Xor,
    }
}

impl Lower<'_> {
    fn is_mask(&self, v: Option<Value>) -> bool {
        v.is_some_and(|v| self.x.st.masks.contains_key(&v))
    }

    pub(super) fn sop(&mut self, inst: &Inst) -> Result<(), Why> {
        let Fields::Sop { dst, src, imm } = &inst.fields else {
            return Err("a scalar ALU instruction without SOP fields");
        };
        let name = inst.name();
        let scc = inst.write(Reg::Scc);
        let dst_code = match dst {
            Some(Operand::Scalar { code, .. }) => Some(*code),
            _ => None,
        };
        let lo = dst_code.and_then(|c| inst.write(Reg::S(c)));
        if inst.write(Reg::S(EXEC)).is_some() || self.is_mask(lo) {
            return Err(LANE_MASK);
        }
        if matches!(name, "s_setreg_imm32_b32" | "s_setreg_b32" | "s_swappc_b64" | "s_version")
            || name.starts_with("s_waitcnt")
        {
            // shortcut: the float MODE register is not modelled; s_swappc_b64 is the
            // vertex fetch call, whose results arrive as inputs.
            return Err("a register the emitter does not model");
        }
        let hi = dst_code.and_then(|c| inst.write(Reg::S(c + 1)));
        let o = Sop { inst, name, dst, src, imm: *imm, c: self.cur, scc, lo, hi };
        if let Some(op) = name.strip_prefix("s_bitcmp") {
            return self.bitcmp(&o, op);
        }
        if let Some(op) = name.strip_prefix("s_cmp_").filter(|o| o.ends_with("_u64")) {
            return self.scalar_cmp64(&o, op);
        }
        if let Some(op) = name.strip_prefix("s_cmp_").or(name.strip_prefix("s_cmpk_")) {
            return self.scalar_cmp(&o, op);
        }
        if name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64") {
            return self.sop64(&o);
        }
        let (r, carry) = match self.sop32_arith(&o)? {
            Some(rc) => rc,
            None => (self.sop32_bits(&o)?, None),
        };
        self.set(lo, Val::I32(r));
        self.sop32_scc(&o, r, carry);
        Ok(())
    }

    /// 32-bit source `k`.
    fn sop_src32(&mut self, o: &Sop, k: usize) -> Result<V, Why> {
        // SOPK arithmetic reads its destination as the first source.
        if k == 0 && matches!(o.name, "s_addk_i32" | "s_mulk_i32") {
            let d = o.dst.as_ref().ok_or("a missing operand")?;
            return self.operand(o.inst, d);
        }
        let op = o.src.get(k).ok_or("a missing operand")?;
        self.operand(o.inst, op)
    }

    /// The SOPK immediate, sign-extended.
    fn sop_sext16(&mut self, o: &Sop) -> V {
        self.s.m.i32(i32::from(o.imm.unwrap_or(0) as i16) as u32)
    }

    /// Dword `h` of 64-bit source `k` (inline constants sign-extend).
    fn sop_half(&mut self, o: &Sop, k: usize, h: u8) -> Result<V, Why> {
        match o.src.get(k) {
            Some(&Operand::Scalar { code, .. }) if code != NULL => {
                let v = o.inst.read(Reg::S(code + h)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            Some(op) if h == 0 => self.operand(o.inst, op),
            Some(Operand::Const(k)) => {
                let x = crate::resource::inline(*k).ok_or("a special operand")?;
                Ok(self.s.m.i32(if (x as i32) < 0 { u32::MAX } else { 0 }))
            }
            _ => Ok(self.s.m.i32(0)),
        }
    }

    /// `s_bitcmp0`/`s_bitcmp1`: SCC is whether bit src1 (mod 32, or mod 64
    /// for the 64-bit form) of src0 is clear (bitcmp0) or set (bitcmp1).
    fn bitcmp(&mut self, o: &Sop, op: &str) -> Result<(), Why> {
        let c = o.c;
        let (bit, k31) = (self.sop_src32(o, 1)?, self.s.m.i32(31));
        let idx = self.s.m.bin(c, Bin::And, bit, k31);
        let word = if op.ends_with("_b64") {
            let (a, b) = (self.sop_half(o, 0, 0)?, self.sop_half(o, 0, 1)?);
            let k32 = self.s.m.i32(32);
            let upper = self.s.m.bin(c, Bin::And, bit, k32);
            let z = self.s.m.i32(0);
            let upper = self.s.m.cmp(c, pred::NE, upper, z);
            self.s.m.select(c, upper, b, a)
        } else {
            self.sop_src32(o, 0)?
        };
        let shifted = self.s.m.bin(c, Bin::LShr, word, idx);
        let one = self.s.m.i32(1);
        let tested = self.s.m.bin(c, Bin::And, shifted, one);
        let want = self.s.m.i32(u32::from(op.starts_with('1')));
        let r = self.s.m.cmp(c, pred::EQ, tested, want);
        self.set(o.scc, Val::Bit(r));
        Ok(())
    }

    /// `s_cmp_eq_u64`/`s_cmp_lg_u64`: eq and lg only, over both dwords.
    fn scalar_cmp64(&mut self, o: &Sop, op: &str) -> Result<(), Why> {
        let c = o.c;
        let mut same = |h| -> Result<V, Why> {
            let (a, b) = (self.sop_half(o, 0, h)?, self.sop_half(o, 1, h)?);
            Ok(self.s.m.cmp(c, pred::EQ, a, b))
        };
        let (lo, hi) = (same(0)?, same(1)?);
        let mut r = self.s.m.bin(c, Bin::And, lo, hi);
        if op.starts_with("lg") {
            let t = self.s.m.bool(true);
            r = self.s.m.bin(c, Bin::Xor, r, t);
        }
        self.set(o.scc, Val::Bit(r));
        Ok(())
    }

    /// `s_cmp_*` and `s_cmpk_*` on 32 bits.
    fn scalar_cmp(&mut self, o: &Sop, op: &str) -> Result<(), Why> {
        let (a, b) = if o.name.starts_with("s_cmpk_") {
            let b = if op.ends_with("_i32") {
                self.sop_sext16(o)
            } else {
                self.s.m.i32(u32::from(o.imm.unwrap_or(0)))
            };
            (self.sop_src32(o, 0)?, b)
        } else {
            if op.ends_with("64") {
                return Err("a 64-bit scalar compare");
            }
            (self.sop_src32(o, 0)?, self.sop_src32(o, 1)?)
        };
        let signed = op.ends_with("_i32");
        let p = match &op[..2] {
            "eq" => pred::EQ,
            "lg" => pred::NE,
            "gt" if signed => pred::SGT,
            "ge" if signed => pred::SGE,
            "lt" if signed => pred::SLT,
            "le" if signed => pred::SLE,
            "gt" => pred::IUGT,
            "ge" => pred::IUGE,
            "lt" => pred::IULT,
            "le" => pred::IULE,
            _ => return Err("a scalar compare"),
        };
        let r = self.s.m.cmp(o.c, p, a, b);
        self.set(o.scc, Val::Bit(r));
        Ok(())
    }

    /// Whether 64-bit `a:b` is not zero.
    fn nonzero64(&mut self, c: usize, a: V, b: V) -> V {
        let o = self.s.m.bin(c, Bin::Or, a, b);
        let z = self.s.m.i32(0);
        self.s.m.cmp(c, pred::NE, o, z)
    }

    /// The 64-bit operations.
    fn sop64(&mut self, o: &Sop) -> Result<(), Why> {
        match o.name {
            "s_bcnt1_i32_b64" | "s_ff1_i32_b64" => self.sop64_count(o),
            "s_and_b64" | "s_or_b64" | "s_xor_b64" | "s_andn2_b64" | "s_orn2_b64"
            | "s_nand_b64" | "s_nor_b64" | "s_xnor_b64" | "s_not_b64" => self.sop64_logic(o),
            "s_lshl_b64" | "s_lshr_b64" | "s_ashr_i64" => self.sop64_shift(o),
            "s_mov_b64" => {
                let (a, b) = (self.sop_half(o, 0, 0)?, self.sop_half(o, 0, 1)?);
                self.set(o.lo, Val::I32(a));
                self.set(o.hi, Val::I32(b));
                Ok(())
            }
            "s_cselect_b64" => {
                let cond = self.scc_in(o.inst)?;
                for (h, v) in [(0, o.lo), (1, o.hi)] {
                    let (a, b) = (self.sop_half(o, 0, h)?, self.sop_half(o, 1, h)?);
                    let r = self.s.m.select(o.c, cond, a, b);
                    self.set(v, Val::I32(r));
                }
                Ok(())
            }
            _ => Err(o.name),
        }
    }

    /// `s_bcnt1_i32_b64` (with SCC) and `s_ff1_i32_b64`.
    fn sop64_count(&mut self, o: &Sop) -> Result<(), Why> {
        let c = o.c;
        let (a, b) = (self.sop_half(o, 0, 0)?, self.sop_half(o, 0, 1)?);
        let i32t = self.s.t.i32;
        if o.name == "s_bcnt1_i32_b64" {
            let na = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[a]);
            let nb = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[b]);
            let r = self.s.m.bin(c, Bin::Add, na, nb);
            self.set(o.lo, Val::I32(r));
            let z = self.s.m.i32(0);
            let s = self.s.m.cmp(c, pred::NE, r, z);
            self.set(o.scc, Val::Bit(s));
            return Ok(());
        }
        // The lowest set bit; -1 for none (as DXIL's first bit low).
        let fa = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[a]);
        let fb = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[b]);
        let (z, k32) = (self.s.m.i32(0), self.s.m.i32(32));
        let fb32 = self.s.m.bin(c, Bin::Add, fb, k32);
        let none = self.s.m.cmp(c, pred::EQ, b, z);
        let fb = self.s.m.select(c, none, fb, fb32);
        let low = self.s.m.cmp(c, pred::NE, a, z);
        let r = self.s.m.select(c, low, fa, fb);
        self.set(o.lo, Val::I32(r));
        Ok(())
    }

    /// The 64-bit bitwise operations, a dword at a time.
    fn sop64_logic(&mut self, o: &Sop) -> Result<(), Why> {
        let (c, name) = (o.c, o.name);
        let ones = self.s.m.i32(u32::MAX);
        let mut r = Vec::new();
        for h in 0..2u8 {
            let a = self.sop_half(o, 0, h)?;
            let x = if name == "s_not_b64" {
                self.s.m.bin(c, Bin::Xor, a, ones)
            } else {
                let mut b = self.sop_half(o, 1, h)?;
                if name.contains("n2") {
                    b = self.s.m.bin(c, Bin::Xor, b, ones);
                }
                let x = self.s.m.bin(c, logic_op(name), a, b);
                if matches!(name, "s_nand_b64" | "s_nor_b64" | "s_xnor_b64") {
                    self.s.m.bin(c, Bin::Xor, x, ones)
                } else {
                    x
                }
            };
            r.push(x);
        }
        self.set(o.lo, Val::I32(r[0]));
        self.set(o.hi, Val::I32(r[1]));
        let s = self.nonzero64(c, r[0], r[1]);
        self.set(o.scc, Val::Bit(s));
        Ok(())
    }

    /// The 64-bit shifts, through a 64-bit integer.
    fn sop64_shift(&mut self, o: &Sop) -> Result<(), Why> {
        let c = o.c;
        let (a, b) = (self.sop_half(o, 0, 0)?, self.sop_half(o, 0, 1)?);
        let n = self.sop_src32(o, 1)?;
        let i64t = self.s.m.int(64);
        self.s.flags |= flags::INT64;
        let a = self.s.m.cast(c, Cast::ZExt, a, i64t);
        let b = self.s.m.cast(c, Cast::ZExt, b, i64t);
        let k32 = self.s.m.int_const(i64t, 32);
        let b = self.s.m.bin(c, Bin::Shl, b, k32);
        let x = self.s.m.bin(c, Bin::Or, a, b);
        let k63 = self.s.m.i32(63);
        let n = self.s.m.bin(c, Bin::And, n, k63);
        let n = self.s.m.cast(c, Cast::ZExt, n, i64t);
        let op = match o.name {
            "s_lshl_b64" => Bin::Shl,
            "s_lshr_b64" => Bin::LShr,
            _ => Bin::AShr,
        };
        let r = self.s.m.bin(c, op, x, n);
        let rl = self.s.m.cast(c, Cast::Trunc, r, self.s.t.i32);
        let rh = self.s.m.bin(c, Bin::LShr, r, k32);
        let rh = self.s.m.cast(c, Cast::Trunc, rh, self.s.t.i32);
        self.set(o.lo, Val::I32(rl));
        self.set(o.hi, Val::I32(rh));
        let s = self.nonzero64(c, rl, rh);
        self.set(o.scc, Val::Bit(s));
        Ok(())
    }

    /// The 32-bit moves and arithmetic: the result and the carry it sets
    /// SCC to, or `None` for an instruction of another family.
    fn sop32_arith(&mut self, o: &Sop) -> Result<Option<(V, Option<V>)>, Why> {
        let (c, name) = (o.c, o.name);
        Ok(Some(match name {
            "s_mov_b32" => (self.sop_src32(o, 0)?, None),
            "s_movk_i32" => (self.sop_sext16(o), None),
            "s_cselect_b32" => {
                let cond = self.scc_in(o.inst)?;
                let (a, b) = (self.sop_src32(o, 0)?, self.sop_src32(o, 1)?);
                (self.s.m.select(c, cond, a, b), None)
            }
            "s_add_u32" | "s_add_i32" | "s_addk_i32" => {
                let a = self.sop_src32(o, 0)?;
                let b =
                    if name == "s_addk_i32" { self.sop_sext16(o) } else { self.sop_src32(o, 1)? };
                let r = self.s.m.bin(c, Bin::Add, a, b);
                let carry = if name == "s_add_u32" {
                    Some(self.s.m.cmp(c, pred::IULT, r, a))
                } else {
                    None
                };
                (r, carry)
            }
            "s_sub_u32" | "s_sub_i32" => {
                let (a, b) = (self.sop_src32(o, 0)?, self.sop_src32(o, 1)?);
                let r = self.s.m.bin(c, Bin::Sub, a, b);
                let carry = (name == "s_sub_u32").then(|| self.s.m.cmp(c, pred::IULT, a, b));
                (r, carry)
            }
            "s_addc_u32" | "s_subb_u32" => {
                let (a, b) = (self.sop_src32(o, 0)?, self.sop_src32(o, 1)?);
                let cin = self.scc_in(o.inst)?;
                let (r, carry) = self.add_sub_carry(c, name == "s_addc_u32", a, b, Some(cin));
                (r, Some(carry))
            }
            "s_mul_i32" | "s_mulk_i32" => {
                let a = self.sop_src32(o, 0)?;
                let b =
                    if name == "s_mulk_i32" { self.sop_sext16(o) } else { self.sop_src32(o, 1)? };
                (self.s.m.bin(c, Bin::Mul, a, b), None)
            }
            "s_mul_hi_u32" | "s_mul_hi_i32" => {
                let (a, b) = (self.sop_src32(o, 0)?, self.sop_src32(o, 1)?);
                (self.mul_hi(a, b, name.ends_with("i32")), None)
            }
            "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
                let (a, b) = (self.sop_src32(o, 0)?, self.sop_src32(o, 1)?);
                let k = self.s.m.i32(u32::from(name.as_bytes()[6] - b'0'));
                let t = self.s.m.bin(c, Bin::Shl, a, k);
                (self.s.m.bin(c, Bin::Add, t, b), None)
            }
            "s_min_u32" | "s_max_u32" | "s_min_i32" | "s_max_i32" => {
                let (a, b) = (self.sop_src32(o, 0)?, self.sop_src32(o, 1)?);
                let p = match name {
                    "s_min_u32" => pred::IULT,
                    "s_max_u32" => pred::IUGT,
                    "s_min_i32" => pred::SLT,
                    _ => pred::SGT,
                };
                let t = self.s.m.cmp(c, p, a, b);
                (self.s.m.select(c, t, a, b), None)
            }
            "s_abs_i32" => {
                let a = self.sop_src32(o, 0)?;
                let z = self.s.m.i32(0);
                let neg = self.s.m.cmp(c, pred::SLT, a, z);
                let n = self.s.m.bin(c, Bin::Sub, z, a);
                (self.s.m.select(c, neg, n, a), None)
            }
            _ => return Ok(None),
        }))
    }

    /// The 32-bit bit operations: counts, field extracts, bitwise logic,
    /// shifts and bit sets.
    fn sop32_bits(&mut self, o: &Sop) -> Result<V, Why> {
        let (c, name) = (o.c, o.name);
        Ok(match name {
            "s_brev_b32" => {
                let a = self.sop_src32(o, 0)?;
                self.s.op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[a])
            }
            "s_not_b32" => {
                let a = self.sop_src32(o, 0)?;
                let ones = self.s.m.i32(u32::MAX);
                self.s.m.bin(c, Bin::Xor, a, ones)
            }
            "s_flbit_i32_b32" | "s_flbit_i32" => {
                // The first bit from the top (differing from the sign for
                // the signed form), -1 for none: DXIL's first-bit-high.
                let a = self.sop_src32(o, 0)?;
                let op = if name == "s_flbit_i32_b32" { 33 } else { 34 };
                self.s.op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a])
            }
            "s_ff1_i32_b32" | "s_bcnt1_i32_b32" => {
                // First set bit (-1 for zero, as DXIL's), or the bit count.
                let a = self.sop_src32(o, 0)?;
                let op = if name == "s_ff1_i32_b32" { 32 } else { 31 };
                self.s.op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a])
            }
            "s_bfe_u32" | "s_bfe_i32" => {
                // src1: offset in bits 0-4, width in bits 16-22.
                let (a, b) = (self.sop_src32(o, 0)?, self.sop_src32(o, 1)?);
                let (k16, k127) = (self.s.m.i32(16), self.s.m.i32(0x7F));
                let w = self.s.m.bin(c, Bin::LShr, b, k16);
                let w = self.s.m.bin(c, Bin::And, w, k127);
                self.bfe(a, b, w, name == "s_bfe_i32")
            }
            "s_and_b32" | "s_or_b32" | "s_xor_b32" | "s_andn2_b32" | "s_orn2_b32"
            | "s_nand_b32" | "s_nor_b32" | "s_xnor_b32" => {
                let (a, mut b) = (self.sop_src32(o, 0)?, self.sop_src32(o, 1)?);
                let ones = self.s.m.i32(u32::MAX);
                if name.contains("n2") {
                    b = self.s.m.bin(c, Bin::Xor, b, ones);
                }
                let mut r = self.s.m.bin(c, logic_op(name), a, b);
                if matches!(name, "s_nand_b32" | "s_nor_b32" | "s_xnor_b32") {
                    r = self.s.m.bin(c, Bin::Xor, r, ones);
                }
                r
            }
            "s_lshl_b32" | "s_lshr_b32" | "s_ashr_i32" => {
                let (a, b) = (self.sop_src32(o, 0)?, self.sop_src32(o, 1)?);
                let k31 = self.s.m.i32(31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let op = match name {
                    "s_lshl_b32" => Bin::Shl,
                    "s_lshr_b32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                self.s.m.bin(c, op, a, b)
            }
            "s_bitset0_b32" | "s_bitset1_b32" => self.bitset(o)?,
            "s_bfm_b32" => {
                let (a, b) = (self.sop_src32(o, 0)?, self.sop_src32(o, 1)?);
                let k31 = self.s.m.i32(31);
                let one = self.s.m.i32(1);
                let a = self.s.m.bin(c, Bin::And, a, k31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let m = self.s.m.bin(c, Bin::Shl, one, a);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                self.s.m.bin(c, Bin::Shl, m, b)
            }
            _ => return Err(name),
        })
    }

    /// `s_bitset0_b32`/`s_bitset1_b32`: sets (clears) bit src[4:0] of the
    /// destination it reads.
    fn bitset(&mut self, o: &Sop) -> Result<V, Why> {
        let c = o.c;
        let old = self.operand(o.inst, o.dst.as_ref().ok_or("a missing operand")?)?;
        let a = self.sop_src32(o, 0)?;
        let (k31, one) = (self.s.m.i32(31), self.s.m.i32(1));
        let a = self.s.m.bin(c, Bin::And, a, k31);
        let bit = self.s.m.bin(c, Bin::Shl, one, a);
        Ok(if o.name == "s_bitset1_b32" {
            self.s.m.bin(c, Bin::Or, old, bit)
        } else {
            let ones = self.s.m.i32(u32::MAX);
            let nb = self.s.m.bin(c, Bin::Xor, bit, ones);
            self.s.m.bin(c, Bin::And, old, nb)
        })
    }

    /// The SCC a 32-bit operation with result `r` sets: its carry, nothing
    /// for moves, selects and multiplies, else whether `r` is not zero.
    fn sop32_scc(&mut self, o: &Sop, r: V, carry: Option<V>) {
        let Some(scc) = o.scc else {
            return;
        };
        let val = match carry {
            Some(k) => Val::Bit(k),
            None if matches!(
                o.name,
                "s_mov_b32"
                    | "s_movk_i32"
                    | "s_cselect_b32"
                    | "s_brev_b32"
                    | "s_mul_i32"
                    | "s_mulk_i32"
            ) =>
            {
                return;
            }
            None if o.name.starts_with("s_add") || o.name.starts_with("s_sub") => {
                Val::Missing("the carry of a scalar operation")
            }
            None => {
                let z = self.s.m.i32(0);
                Val::Bit(self.s.m.cmp(o.c, pred::NE, r, z))
            }
        };
        self.vals.insert(scc, val);
    }

    /// `a + b (+ cin)` or `a - b (- cin)` with the carry (borrow) out, as the
    /// scalar and vector add-with-carry instructions set it.
    pub(super) fn add_sub_carry(
        &mut self,
        c: usize,
        add: bool,
        a: V,
        b: V,
        cin: Option<V>,
    ) -> (V, V) {
        let op = if add { Bin::Add } else { Bin::Sub };
        let mut r = self.s.m.bin(c, op, a, b);
        // Without a carry in: a + b wraps below a; a - b borrows when a < b.
        let mut carry =
            if add { self.s.m.cmp(c, pred::IULT, r, a) } else { self.s.m.cmp(c, pred::IULT, a, b) };
        if let Some(cin) = cin {
            let one = self.s.m.cast(c, Cast::ZExt, cin, self.s.t.i32);
            let t = r;
            r = self.s.m.bin(c, op, t, one);
            // The carry in carries on when the partial result is all ones
            // (add) or zero (sub).
            let edge = self.s.m.i32(if add { u32::MAX } else { 0 });
            let at_edge = self.s.m.cmp(c, pred::EQ, t, edge);
            let more = self.s.m.bin(c, Bin::And, cin, at_edge);
            carry = self.s.m.bin(c, Bin::Or, carry, more);
        }
        (r, carry)
    }

    fn scc_in(&mut self, inst: &Inst) -> Result<V, Why> {
        let v = inst.read(Reg::Scc).ok_or("an unread SCC")?;
        match self.get(v)? {
            Val::Bit(x) => Ok(x),
            _ => Err("SCC without a value"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{S_ENDPGM, lowers};

    #[test]
    fn a_scalar_carry_chain_reaches_the_select() {
        // The high dword of a 64-bit add or subtract, then a select on the
        // carry (borrow) the high half leaves in SCC, and a branch on it
        // (only code with an effect needs its operands).
        let add = [
            0xBE80_03C1, // s_mov_b32 s0, -1
            0xBE81_0380, // s_mov_b32 s1, 0
            0x8000_8100, // s_add_u32 s0, s0, 1
            0x8201_8001, // s_addc_u32 s1, s1, 0
            0x8502_0001, // s_cselect_b32 s2, s1, s0
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE83_0380, // s_mov_b32 s3, 0
            S_ENDPGM,
        ];
        let sub = [
            0xBE80_0380, // s_mov_b32 s0, 0
            0xBE81_0380, // s_mov_b32 s1, 0
            0x8080_8100, // s_sub_u32 s0, s0, 1
            0x8281_8001, // s_subb_u32 s1, s1, 0
            0x8502_0001, // s_cselect_b32 s2, s1, s0
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE83_0380, // s_mov_b32 s3, 0
            S_ENDPGM,
        ];

        for words in [add, sub] {
            lowers(&words, 0).unwrap();
        }
    }

    #[test]
    fn scalar_bit_tests_lower() {
        // Each tests one bit of a constant into SCC and branches on it (only
        // code with an effect needs its operands).
        let tail = [
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        let bit_tests: [&[u32]; 4] = [
            &[0xBE81_0387, 0xBF0C_8301], // s_mov_b32 s1, 7; s_bitcmp0_b32 s1, 3
            &[0xBE81_0387, 0xBF0D_8301], // s_mov_b32 s1, 7; s_bitcmp1_b32 s1, 3
            &[
                0xBE80_0387, // s_mov_b32 s0, 7
                0xBE81_0380, // s_mov_b32 s1, 0
                0xBF0E_8300, // s_bitcmp0_b64 s[0:1], 3
            ],
            &[
                0xBE80_0387, // s_mov_b32 s0, 7
                0xBE81_0380, // s_mov_b32 s1, 0
                0xBF0F_8300, // s_bitcmp1_b64 s[0:1], 3
            ],
        ];
        for head in bit_tests {
            let words: Vec<u32> = head.iter().chain(&tail).copied().collect();
            lowers(&words, 0).unwrap();
        }
    }
}

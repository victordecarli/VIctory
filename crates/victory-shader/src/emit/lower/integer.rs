//! 32-bit vector integer operations (bitwise, shifts, multiplies, field
//! and byte operations, lane counts) and the vector compares.

use super::*;

impl Lower<'_> {
    /// `x & 31`.
    fn mask31(&mut self, c: usize, x: V) -> V {
        let k31 = self.s.m.i32(31);
        self.s.m.bin(c, Bin::And, x, k31)
    }

    pub(super) fn integer(&mut self, inst: &Inst, name: &'static str) -> Result<V, Why> {
        let c = self.cur;
        if let Some(r) = self.int_logic(inst, name, c)? {
            return Ok(r);
        }
        if let Some(r) = self.int_fields(inst, name, c)? {
            return Ok(r);
        }
        if let Some(r) = self.int_arith(inst, name, c)? {
            return Ok(r);
        }
        match name {
            "v_perm_b32" => self.perm(inst, name, c),
            "v_cvt_pk_u8_f32" => self.cvt_pk_u8(inst, c),
            "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => self.mbcnt(inst, name, c),
            _ => Err(name),
        }
    }

    /// Bitwise operations, shifts, and the two-operation forms; `None` for
    /// another family.
    fn int_logic(&mut self, inst: &Inst, name: &str, c: usize) -> Result<Option<V>, Why> {
        Ok(Some(match name {
            "v_and_b32" | "v_or_b32" | "v_xor_b32" | "v_add_nc_u32" | "v_sub_nc_u32"
            | "v_subrev_nc_u32" | "v_mul_lo_u32" => {
                let (a, b) = (self.src(inst, 0)?, self.src(inst, 1)?);
                let (op, a, b) = match name {
                    "v_and_b32" => (Bin::And, a, b),
                    "v_or_b32" => (Bin::Or, a, b),
                    "v_xor_b32" => (Bin::Xor, a, b),
                    "v_add_nc_u32" => (Bin::Add, a, b),
                    "v_sub_nc_u32" => (Bin::Sub, a, b),
                    "v_subrev_nc_u32" => (Bin::Sub, b, a),
                    _ => (Bin::Mul, a, b),
                };
                self.s.m.bin(c, op, a, b)
            }
            "v_xnor_b32" => {
                let (a, b) = (self.src(inst, 0)?, self.src(inst, 1)?);
                let x = self.s.m.bin(c, Bin::Xor, a, b);
                let ones = self.s.m.i32(u32::MAX);
                self.s.m.bin(c, Bin::Xor, x, ones)
            }
            "v_not_b32" => {
                let a = self.src(inst, 0)?;
                let ones = self.s.m.i32(u32::MAX);
                self.s.m.bin(c, Bin::Xor, a, ones)
            }
            "v_lshlrev_b32" | "v_lshrrev_b32" | "v_ashrrev_i32" => {
                let (sh, a) = (self.src(inst, 0)?, self.src(inst, 1)?);
                let sh = self.mask31(c, sh);
                let op = match name {
                    "v_lshlrev_b32" => Bin::Shl,
                    "v_lshrrev_b32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                self.s.m.bin(c, op, a, sh)
            }
            "v_lshl_add_u32" | "v_lshl_or_b32" | "v_add_lshl_u32" | "v_and_or_b32"
            | "v_or3_b32" | "v_xad_u32" | "v_add3_u32" | "v_xor3_b32" => {
                let (a, b, d) = (self.src(inst, 0)?, self.src(inst, 1)?, self.src(inst, 2)?);
                match name {
                    "v_lshl_add_u32" | "v_lshl_or_b32" => {
                        let b = self.mask31(c, b);
                        let t = self.s.m.bin(c, Bin::Shl, a, b);
                        let op = if name == "v_lshl_add_u32" { Bin::Add } else { Bin::Or };
                        self.s.m.bin(c, op, t, d)
                    }
                    "v_add_lshl_u32" => {
                        let t = self.s.m.bin(c, Bin::Add, a, b);
                        let d = self.mask31(c, d);
                        self.s.m.bin(c, Bin::Shl, t, d)
                    }
                    _ => {
                        let (o1, o2) = match name {
                            "v_and_or_b32" => (Bin::And, Bin::Or),
                            "v_or3_b32" => (Bin::Or, Bin::Or),
                            "v_xad_u32" => (Bin::Xor, Bin::Add),
                            "v_add3_u32" => (Bin::Add, Bin::Add),
                            _ => (Bin::Xor, Bin::Xor),
                        };
                        let t = self.s.m.bin(c, o1, a, b);
                        self.s.m.bin(c, o2, t, d)
                    }
                }
            }
            _ => return Ok(None),
        }))
    }

    /// Bit fields, bit counts and reversal, and `v_alignbit`/`v_alignbyte`;
    /// `None` for another family.
    fn int_fields(&mut self, inst: &Inst, name: &str, c: usize) -> Result<Option<V>, Why> {
        Ok(Some(match name {
            "v_bfe_u32" | "v_bfe_i32" => {
                let (a, off, w) = (self.src(inst, 0)?, self.src(inst, 1)?, self.src(inst, 2)?);
                let w = self.mask31(c, w);
                self.bfe(a, off, w, name == "v_bfe_i32")
            }
            "v_bfi_b32" => {
                let (m, a, b) = (self.src(inst, 0)?, self.src(inst, 1)?, self.src(inst, 2)?);
                let ones = self.s.m.i32(u32::MAX);
                let nm = self.s.m.bin(c, Bin::Xor, m, ones);
                let x = self.s.m.bin(c, Bin::And, m, a);
                let y = self.s.m.bin(c, Bin::And, nm, b);
                self.s.m.bin(c, Bin::Or, x, y)
            }
            "v_bfm_b32" => {
                // ((1 << src0[4:0]) - 1) << src1[4:0]
                let (a, b) = (self.src(inst, 0)?, self.src(inst, 1)?);
                let one = self.s.m.i32(1);
                let (a, b) = (self.mask31(c, a), self.mask31(c, b));
                let m = self.s.m.bin(c, Bin::Shl, one, a);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                self.s.m.bin(c, Bin::Shl, m, b)
            }
            "v_alignbit_b32" | "v_alignbyte_b32" => {
                // The low dword of {a, b} >> shift.
                let (a, b, sh) = (self.src(inst, 0)?, self.src(inst, 1)?, self.src(inst, 2)?);
                let sh = if name == "v_alignbyte_b32" {
                    let three = self.s.m.i32(3);
                    let t = self.s.m.bin(c, Bin::And, sh, three);
                    let k3 = self.s.m.i32(3);
                    self.s.m.bin(c, Bin::Shl, t, k3)
                } else {
                    self.mask31(c, sh)
                };
                let k32 = self.s.m.i32(32);
                let lo = self.s.m.bin(c, Bin::LShr, b, sh);
                let left = self.s.m.bin(c, Bin::Sub, k32, sh);
                let left = self.mask31(c, left);
                let hi = self.s.m.bin(c, Bin::Shl, a, left);
                let zero = self.s.m.i32(0);
                let none = self.s.m.cmp(c, pred::EQ, sh, zero);
                let hi = self.s.m.select(c, none, zero, hi);
                self.s.m.bin(c, Bin::Or, lo, hi)
            }
            "v_bfrev_b32" => {
                let a = self.src(inst, 0)?;
                self.s.op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[a])
            }
            "v_bcnt_u32_b32" => {
                let (a, b) = (self.src(inst, 0)?, self.src(inst, 1)?);
                let n = self.s.op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 31, &[a]);
                self.s.m.bin(c, Bin::Add, n, b)
            }
            "v_ffbl_b32" => {
                // -1 for zero, as DXIL's first-bit-low.
                let a = self.src(inst, 0)?;
                self.s.op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 32, &[a])
            }
            _ => return Ok(None),
        }))
    }

    /// 24-bit multiplies and multiply-adds, the high half of a multiply,
    /// and the two- and three-way minimum, maximum and median; `None` for
    /// another family.
    fn int_arith(&mut self, inst: &Inst, name: &str, c: usize) -> Result<Option<V>, Why> {
        Ok(Some(match name {
            "v_mul_u32_u24" | "v_mul_i32_i24" | "v_mad_u32_u24" | "v_mad_i32_i24" => {
                let (a, b) = (self.src(inst, 0)?, self.src(inst, 1)?);
                let d = if name.starts_with("v_mad") { Some(self.src(inst, 2)?) } else { None };
                let (sh, op) =
                    (self.s.m.i32(8), if name.ends_with("i24") { Bin::AShr } else { Bin::LShr });
                let a = self.s.m.bin(c, Bin::Shl, a, sh);
                let a = self.s.m.bin(c, op, a, sh);
                let b = self.s.m.bin(c, Bin::Shl, b, sh);
                let b = self.s.m.bin(c, op, b, sh);
                let m = self.s.m.bin(c, Bin::Mul, a, b);
                match d {
                    Some(d) => self.s.m.bin(c, Bin::Add, m, d),
                    None => m,
                }
            }
            "v_max_i32" | "v_min_i32" | "v_max_u32" | "v_min_u32" => {
                let (a, b) = (self.src(inst, 0)?, self.src(inst, 1)?);
                let p = match name {
                    "v_max_i32" => pred::SGT,
                    "v_min_i32" => pred::SLT,
                    "v_max_u32" => pred::IUGT,
                    _ => pred::IULT,
                };
                let t = self.s.m.cmp(c, p, a, b);
                self.s.m.select(c, t, a, b)
            }
            "v_mul_hi_u32" | "v_mul_hi_i32" => {
                let (a, b) = (self.src(inst, 0)?, self.src(inst, 1)?);
                self.mul_hi(a, b, name == "v_mul_hi_i32")
            }
            "v_min3_i32" | "v_max3_i32" | "v_min3_u32" | "v_max3_u32" | "v_med3_i32"
            | "v_med3_u32" => {
                let (a, b, d) = (self.src(inst, 0)?, self.src(inst, 1)?, self.src(inst, 2)?);
                let signed = name.ends_with("i32");
                let (lt, gt) =
                    if signed { (pred::SLT, pred::SGT) } else { (pred::IULT, pred::IUGT) };
                let pick = |l: &mut Self, p, x, y| {
                    let t = l.s.m.cmp(c, p, x, y);
                    l.s.m.select(c, t, x, y)
                };
                match &name[2..6] {
                    "min3" => {
                        let t = pick(self, lt, a, b);
                        pick(self, lt, t, d)
                    }
                    "max3" => {
                        let t = pick(self, gt, a, b);
                        pick(self, gt, t, d)
                    }
                    _ => {
                        let lo = pick(self, lt, a, b);
                        let hi = pick(self, gt, a, b);
                        let t = pick(self, lt, hi, d);
                        pick(self, gt, lo, t)
                    }
                }
            }
            _ => return Ok(None),
        }))
    }

    /// `v_perm_b32`: byte i of the result is byte S of {src0, src1} (src1
    /// the low dword) for selector byte S below 8; 8-11 repeat the sign of
    /// bytes 1 and 3 of src1, then of src0; 12 is 0x00, higher 0xFF.
    fn perm(&mut self, inst: &Inst, name: &'static str, c: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            return Err("a vector ALU instruction without VALU fields");
        };
        let sel = match v.src.get(2) {
            Some(&Operand::Literal(x)) => x,
            Some(&Operand::Const(k)) => crate::resource::inline(k).ok_or(name)?,
            _ => return self.perm_dynamic(inst, c),
        };
        let (a, b) = (self.src(inst, 0)?, self.src(inst, 1)?);
        let mut r = self.s.m.i32(0);
        for i in 0..4 {
            let pick = sel >> (8 * i) & 0xFF;
            let byte = match pick {
                0..=7 => {
                    let (src, n) = if pick < 4 { (b, pick) } else { (a, pick - 4) };
                    let sh = self.s.m.i32(8 * n);
                    let t = self.s.m.bin(c, Bin::LShr, src, sh);
                    let ff = self.s.m.i32(0xFF);
                    self.s.m.bin(c, Bin::And, t, ff)
                }
                8..=11 => {
                    let src = if pick < 10 { b } else { a };
                    let bit = if pick % 2 == 0 { 15 } else { 31 };
                    let left = self.s.m.i32(31 - bit);
                    let t = self.s.m.bin(c, Bin::Shl, src, left);
                    let k31 = self.s.m.i32(31);
                    let t = self.s.m.bin(c, Bin::AShr, t, k31);
                    let ff = self.s.m.i32(0xFF);
                    self.s.m.bin(c, Bin::And, t, ff)
                }
                12 => continue,
                _ => self.s.m.i32(0xFF),
            };
            let at = self.s.m.i32(8 * i);
            let byte = self.s.m.bin(c, Bin::Shl, byte, at);
            r = self.s.m.bin(c, Bin::Or, r, byte);
        }
        Ok(r)
    }

    /// `v_perm_b32` with a computed selector: each byte picked at run time.
    fn perm_dynamic(&mut self, inst: &Inst, c: usize) -> Result<V, Why> {
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let (a, b, sv) = (self.src(inst, 0)?, self.src(inst, 1)?, self.src(inst, 2)?);
        let mut r = k(self, 0);
        for i in 0..4u32 {
            let (sh, ff) = (k(self, 8 * i), k(self, 0xFF));
            let sel = self.s.m.bin(c, Bin::LShr, sv, sh);
            let sel = self.s.m.bin(c, Bin::And, sel, ff);
            // Select a dword and a byte within it.
            // Later selects handle codes 8 and above.
            let four = k(self, 4);
            let zero = k(self, 0);
            let high = self.s.m.bin(c, Bin::And, sel, four);
            let high = self.s.m.cmp(c, pred::NE, high, zero);
            let word = self.s.m.select(c, high, a, b);
            let three = k(self, 3);
            let index = self.s.m.bin(c, Bin::And, sel, three);
            let bits = self.s.m.bin(c, Bin::Shl, index, three);
            let x = self.s.m.bin(c, Bin::LShr, word, bits);
            let mut byte = self.s.m.bin(c, Bin::And, x, ff);
            // 8-11: the sign of bytes 1, 3 of src1, then of src0.
            for (n, src, bit) in [(8, b, 15), (9, b, 31), (10, a, 15), (11, a, 31)] {
                let left = k(self, 31 - bit);
                let t = self.s.m.bin(c, Bin::Shl, src, left);
                let k31 = k(self, 31);
                let t = self.s.m.bin(c, Bin::AShr, t, k31);
                let t = self.s.m.bin(c, Bin::And, t, ff);
                let kn = k(self, n);
                let is = self.s.m.cmp(c, pred::EQ, sel, kn);
                byte = self.s.m.select(c, is, t, byte);
            }
            let (k12, z) = (k(self, 12), k(self, 0));
            let is12 = self.s.m.cmp(c, pred::EQ, sel, k12);
            byte = self.s.m.select(c, is12, z, byte);
            let above = self.s.m.cmp(c, pred::IUGT, sel, k12);
            byte = self.s.m.select(c, above, ff, byte);
            let at = k(self, 8 * i);
            let byte = self.s.m.bin(c, Bin::Shl, byte, at);
            r = self.s.m.bin(c, Bin::Or, r, byte);
        }
        Ok(r)
    }

    /// `v_cvt_pk_u8_f32`: src0 rounded to nearest even and saturated to a
    /// byte, into byte src1[1:0] of src2.
    fn cvt_pk_u8(&mut self, inst: &Inst, c: usize) -> Result<V, Why> {
        let f = self.fsrc(inst, 0)?;
        let (b, d) = (self.src(inst, 1)?, self.src(inst, 2)?);
        let f = self.unary(26, f);
        let (zero, max) = (self.s.m.f32(0), self.s.m.f32(255.0f32.to_bits()));
        let f = self.binary(35, f, zero);
        let f = self.binary(36, f, max);
        let x = self.s.m.cast(c, Cast::FpToUi, f, self.s.t.i32);
        let (three, k3) = (self.s.m.i32(3), self.s.m.i32(3));
        let sh = self.s.m.bin(c, Bin::And, b, three);
        let sh = self.s.m.bin(c, Bin::Shl, sh, k3);
        let x = self.s.m.bin(c, Bin::Shl, x, sh);
        let ff = self.s.m.i32(0xFF);
        let hole = self.s.m.bin(c, Bin::Shl, ff, sh);
        let ones = self.s.m.i32(u32::MAX);
        let hole = self.s.m.bin(c, Bin::Xor, hole, ones);
        let d = self.s.m.bin(c, Bin::And, d, hole);
        Ok(self.s.m.bin(c, Bin::Or, d, x))
    }

    /// `v_mbcnt_lo_u32_b32` and `v_mbcnt_hi_u32_b32`: bits of the mask's
    /// low (high) half below this lane, plus src1: with an all-ones mask,
    /// the lane index.
    fn mbcnt(&mut self, inst: &Inst, name: &str, c: usize) -> Result<V, Why> {
        let (a, b) = (self.src(inst, 0)?, self.src(inst, 1)?);
        // Mesh lanes are the console's by construction; pixel and
        // vertex lanes are the host's, as their masks are.
        if self.mask_data == MaskData::Host {
            // Host lanes: the mask must be one (a ballot) or all lanes.
            let Fields::Valu(v) = &inst.fields else {
                return Err("a vector ALU instruction without VALU fields");
            };
            let host = match v.src.first() {
                Some(&Operand::Const(193)) => true,
                Some(&Operand::Scalar { code, .. }) => inst.read(Reg::S(code)).is_some_and(|m| {
                    self.x.st.masks.contains_key(&m) || self.x.st.highs.contains_key(&m)
                }),
                _ => false,
            };
            if !host {
                return Err("v_mbcnt of a mask that is no ballot (console lane layout)");
            }
            self.s.flags |= flags::WAVE_OPS;
        } else if self.s.mesh.is_none() {
            self.cross_lane(false)?;
        }
        let lane = self.lane_index();
        let below = if name == "v_mbcnt_lo_u32_b32" {
            lane
        } else {
            let (k32, z) = (self.s.m.i32(32), self.s.m.i32(0));
            let t = self.s.m.bin(c, Bin::Sub, lane, k32);
            let low = self.s.m.cmp(c, pred::IULT, lane, k32);
            self.s.m.select(c, low, z, t)
        };
        // (1 << below) - 1, all ones from 32 on.
        let (one, k32, ones) = (self.s.m.i32(1), self.s.m.i32(32), self.s.m.i32(u32::MAX));
        let sh = self.mask31(c, below);
        let m = self.s.m.bin(c, Bin::Shl, one, sh);
        let m = self.s.m.bin(c, Bin::Sub, m, one);
        let full = self.s.m.cmp(c, pred::IUGE, below, k32);
        let m = self.s.m.select(c, full, ones, m);
        let x = self.s.m.bin(c, Bin::And, a, m);
        let n = self.s.op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 31, &[x]);
        Ok(self.s.m.bin(c, Bin::Add, n, b))
    }

    pub(super) fn compare(&mut self, inst: &Inst, rest: &str) -> Result<(), Why> {
        let (op, ty) = rest.rsplit_once('_').ok_or("a compare")?;
        let c = self.cur;
        let r = match ty {
            "f32" | "f16" if op == "class" => self.compare_class(inst, ty == "f16", c)?,
            "f32" | "f16" => self.compare_float(inst, op, ty, c)?,
            "u64" | "i64" => {
                let (a, b) = (self.src64(inst, 0)?, self.src64(inst, 1)?);
                let signed = ty == "i64";
                let p = match op {
                    "eq" => pred::EQ,
                    "ne" | "lg" => pred::NE,
                    "lt" if signed => pred::SLT,
                    "le" if signed => pred::SLE,
                    "gt" if signed => pred::SGT,
                    "ge" if signed => pred::SGE,
                    "lt" => pred::IULT,
                    "le" => pred::IULE,
                    "gt" => pred::IUGT,
                    "ge" => pred::IUGE,
                    _ => return Err("a 64-bit compare"),
                };
                self.s.m.cmp(c, p, a, b)
            }
            "u32" | "i32" | "u16" | "i16" => {
                let (a, b) = if ty.ends_with("16") {
                    let (mut a, mut b) = (self.half_src(inst, 0)?, self.half_src(inst, 1)?);
                    if ty == "i16" {
                        let k16 = self.s.m.i32(16);
                        for x in [&mut a, &mut b] {
                            let t = self.s.m.bin(c, Bin::Shl, *x, k16);
                            *x = self.s.m.bin(c, Bin::AShr, t, k16);
                        }
                    }
                    (a, b)
                } else {
                    (self.src(inst, 0)?, self.src(inst, 1)?)
                };
                let signed = ty == "i32" || ty == "i16";
                let p = match op {
                    "eq" => pred::EQ,
                    "ne" | "lg" => pred::NE,
                    "lt" if signed => pred::SLT,
                    "le" if signed => pred::SLE,
                    "gt" if signed => pred::SGT,
                    "ge" if signed => pred::SGE,
                    "lt" => pred::IULT,
                    "le" => pred::IULE,
                    "gt" => pred::IUGT,
                    "ge" => pred::IUGE,
                    "f" | "t" => {
                        self.lane(inst, Some(op == "t"));
                        return Ok(());
                    }
                    _ => return Err("an integer compare"),
                };
                self.s.m.cmp(c, p, a, b)
            }
            _ => return Err(inst.name()),
        };
        self.lane_bits_for(inst, r);
        Ok(())
    }

    /// `v_cmp_class_f32`/`_f16`: bit k of src1 tests class k: signalling
    /// NaN, quiet NaN, -inf, -normal, -denormal, -0, +0, +denormal,
    /// +normal, +inf.
    fn compare_class(&mut self, inst: &Inst, f16: bool, c: usize) -> Result<V, Why> {
        let x = if f16 {
            let Fields::Valu(v) = &inst.fields else {
                return Err("a vector ALU instruction without VALU fields");
            };
            let mut h = self.half_src(inst, 0)?;
            if v.abs & 1 != 0 {
                let m = self.s.m.i32(0x7FFF);
                h = self.s.m.bin(c, Bin::And, h, m);
            }
            if v.neg & 1 != 0 {
                let s = self.s.m.i32(0x8000);
                h = self.s.m.bin(c, Bin::Xor, h, s);
            }
            h
        } else {
            let x = self.fsrc(inst, 0)?;
            self.as_int(x)
        };
        let mask = self.src(inst, 1)?;
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let (exp_sh, sign_sh, exp_mask, mant_mask, quiet_bit) =
            if f16 { (10, 15, 0x1F, 0x03FF, 0x0200) } else { (23, 31, 0xFF, 0x7F_FFFF, 0x40_0000) };
        let (k_exp, k_sign, k_emax, mant, quiet) = (
            k(self, exp_sh),
            k(self, sign_sh),
            k(self, exp_mask),
            k(self, mant_mask),
            k(self, quiet_bit),
        );
        let e = self.s.m.bin(c, Bin::LShr, x, k_exp);
        let e = self.s.m.bin(c, Bin::And, e, k_emax);
        let m = self.s.m.bin(c, Bin::And, x, mant);
        let neg = self.s.m.bin(c, Bin::LShr, x, k_sign);
        let zero = k(self, 0);
        let (e0, emax) = (self.s.m.cmp(c, pred::EQ, e, zero), self.s.m.cmp(c, pred::EQ, e, k_emax));
        let m0 = self.s.m.cmp(c, pred::EQ, m, zero);
        let q = self.s.m.bin(c, Bin::And, x, quiet);
        let q = self.s.m.cmp(c, pred::NE, q, zero);
        // Positive classes 6-9; negative ones are 11 minus those.
        let (normal, inf, zero_c, denorm) = (k(self, 8), k(self, 9), k(self, 6), k(self, 7));
        let t = self.s.m.select(c, m0, zero_c, denorm);
        let t = self.s.m.select(c, e0, t, normal);
        let t = self.s.m.select(c, emax, inf, t);
        let eleven = k(self, 11);
        let mirrored = self.s.m.bin(c, Bin::Sub, eleven, t);
        let is_neg = self.s.m.cmp(c, pred::NE, neg, zero);
        let t = self.s.m.select(c, is_neg, mirrored, t);
        let (one, snan) = (k(self, 1), k(self, 0));
        let nan_class = self.s.m.select(c, q, one, snan);
        let yes = self.s.m.bool(true);
        let nan = self.s.m.bin(c, Bin::Xor, m0, yes);
        let nan = self.s.m.bin(c, Bin::And, nan, emax);
        let t = self.s.m.select(c, nan, nan_class, t);
        let bit = self.s.m.bin(c, Bin::LShr, mask, t);
        let bit = self.s.m.bin(c, Bin::And, bit, one);
        Ok(self.s.m.cmp(c, pred::NE, bit, zero))
    }

    /// The f32 and f16 compares other than class.
    fn compare_float(&mut self, inst: &Inst, op: &str, ty: &str, c: usize) -> Result<V, Why> {
        let (a, b) = if ty == "f16" {
            (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?)
        } else {
            (self.fsrc(inst, 0)?, self.fsrc(inst, 1)?)
        };
        if op == "lg" {
            // Ordered and not equal, false for a NaN, as `olt | ogt`.
            // `fcmp one` means the same, but the AMD D3D12 driver
            // runs it as unordered (true for a NaN) where NVIDIA
            // and WARP do not; olt and ogt agree on all three.
            let lt = self.s.m.cmp(c, pred::OLT, a, b);
            let gt = self.s.m.cmp(c, pred::OGT, a, b);
            return Ok(self.s.m.bin(c, Bin::Or, lt, gt));
        }
        let p = match op {
            "f" => pred::FALSE,
            "lt" => pred::OLT,
            "eq" => pred::OEQ,
            "le" => pred::OLE,
            "gt" => pred::OGT,
            "ge" => pred::OGE,
            "o" => pred::ORD,
            "u" => pred::UNO,
            "nge" => pred::ULT,
            "nlg" => pred::UEQ,
            "ngt" => pred::ULE,
            "nle" => pred::UGT,
            "neq" => pred::UNE,
            "nlt" => pred::UGE,
            "tru" => pred::TRUE,
            _ => return Err(inst.name()),
        };
        Ok(self.s.m.cmp(c, p, a, b))
    }
}

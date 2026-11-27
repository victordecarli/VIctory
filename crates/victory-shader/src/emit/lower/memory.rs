//! Vector memory: buffer loads, stores and atomics (typed ones through their
//! buffer format), and global loads from a root SRV or a code table.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Numeric {
    Unorm,
    Snorm,
    Uscaled,
    Sscaled,
    Uint,
    Sint,
    Float,
}

/// A GFX10 buffer format (the unified format of typed buffer loads): the
/// channel widths from the lowest bits up, and how the channels convert.
/// Formats are numbered in families (`8`, `16`, `8_8`, ...: AMD names
/// list channels from the highest bits), each with its numeric formats in
/// the order unorm, snorm, uscaled, sscaled, uint, sint, float.
fn buffer_format(format: u8) -> Option<(&'static [u32], Numeric)> {
    use Numeric::*;
    const NORM: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint];
    const ALL: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint, Float];
    const WIDE: &[Numeric] = &[Uint, Sint, Float];
    // Numbers and order from LLVM's `UfmtGFX10` (`SIDefines.h`); the GFX10
    // buffer formats have no 5_9_9_9. Channel order from Mesa
    // (`ac_formats.c`): R11G11B10_FLOAT is 10_11_11, and 10_10_10_2 has
    // its 2-bit channel in the lowest bits.
    const FAMILIES: [(&[u32], &[Numeric]); 14] = [
        (&[8], NORM),
        (&[16], ALL),
        (&[8, 8], NORM),
        (&[32], WIDE),
        (&[16, 16], ALL),
        (&[11, 11, 10], ALL),     // 10_11_11
        (&[10, 11, 11], ALL),     // 11_11_10
        (&[2, 10, 10, 10], NORM), // 10_10_10_2
        (&[10, 10, 10, 2], NORM), // 2_10_10_10
        (&[8, 8, 8, 8], NORM),
        (&[32, 32], WIDE),
        (&[16, 16, 16, 16], ALL),
        (&[32, 32, 32], WIDE),
        (&[32, 32, 32, 32], WIDE),
    ];
    let mut n = usize::from(format.checked_sub(1)?);
    for (bits, numerics) in FAMILIES {
        if n < numerics.len() {
            return Some((bits, numerics[n]));
        }
        n -= numerics.len();
    }
    None
}

/// How far a float channel of `width` bits (16, or the unsigned 11- and
/// 10-bit floats of the packed formats) shifts up to be a half:
/// the small floats are a half's exponent (5 bits, biased 15) and the top
/// of its mantissa, without the sign, so `legacyF16ToF32` widens them
/// exactly, denormals, infinities and NaNs included.
fn half_shift(width: u32) -> u32 {
    debug_assert!(matches!(width, 10 | 11 | 16), "{width}-bit float");
    if width == 16 { 0 } else { 15 - width }
}

/// DXIL's atomic operation for a `buffer_atomic_*` instruction: add, and,
/// or, xor, signed and unsigned min and max, exchange; compare-exchange
/// apart (`u32::MAX`). `None` for an access that is no atomic.
fn atomic_op(name: &'static str) -> Result<Option<u32>, Why> {
    Ok(match name.strip_prefix("buffer_atomic_") {
        None => None,
        Some("add") => Some(0),
        Some("and") => Some(1),
        Some("or") => Some(2),
        Some("xor") => Some(3),
        Some("smin") => Some(4),
        Some("smax") => Some(5),
        Some("umin") => Some(6),
        Some("umax") => Some(7),
        Some("swap") => Some(8),
        Some("cmpswap") => Some(u32::MAX),
        Some(_) => return Err(name),
    })
}

/// Whether a buffer access is typed, and the dwords or channels it moves
/// (an atomic: one).
fn shape(name: &str, atomic: bool) -> Option<(bool, u32)> {
    let plain = name.replace("_format_d16_", "_format_");
    Some(
        match plain
            .strip_prefix("buffer_load_")
            .or(plain.strip_prefix("buffer_store_"))
            .or(plain.strip_prefix("tbuffer_load_"))
        {
            _ if atomic => (false, 1),
            Some("dword") => (false, 1),
            Some("dwordx2") => (false, 2),
            Some("dwordx3") => (false, 3),
            Some("dwordx4") => (false, 4),
            Some("format_x") => (true, 1),
            Some("format_xy") => (true, 2),
            Some("format_xyz") => (true, 3),
            Some("format_xyzw") => (true, 4),
            _ => return None,
        },
    )
}

/// What the parts of one buffer instruction share.
struct Access<'a> {
    site: Site,
    guard: Option<V>,
    /// The block the instruction starts in.
    c: usize,
    data: Option<Operand>,
    offset: u16,
    offen: bool,
    format: Option<u8>,
    glc: bool,
    store: bool,
    /// The DXIL atomic operation ([`atomic_op`]).
    atomic: Option<u32>,
    d16: bool,
    typed: bool,
    /// Dwords or channels moved.
    count: u32,
    /// The element index (`idxen`).
    index: Option<V>,
    /// The byte offset: the instruction's, its VGPR's and SGPR's, and the
    /// descriptor's when code built it.
    off: V,
    src: &'a Source,
    used: Option<&'a UsedResource>,
}

impl<'a> Lower<'a> {
    /// A global memory load (`global_load_*`): from the program's constant
    /// table, or from a root SRV at the address the title sets.
    pub(super) fn flat(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat { dst, .. } = &inst.fields else {
            return Err("a global instruction without flat fields");
        };
        let name = inst.name();
        let rest = name.strip_prefix("global_load_").ok_or("a global memory write")?;
        let (n, sub) = match rest {
            "dword" => (1, None),
            "dwordx2" => (2, None),
            "dwordx3" => (3, None),
            "dwordx4" => (4, None),
            "ubyte" => (1, Some((8, false))),
            "sbyte" => (1, Some((8, true))),
            "ushort" => (1, Some((16, false))),
            "sshort" => (1, Some((16, true))),
            _ => return Err(name),
        };
        let a = *self.scalar.get(&(b, i)).ok_or("an untracked global address")?;
        let guard = self.guard(b, i)?;
        let Some(Operand::Vector { reg, .. }) = *dst else {
            return Err("a global load result");
        };
        let c = self.cur;
        let mut off = self.s.m.i32(a.offset);
        if let Some(d) = a.dynamic {
            let d = self.i32_of(d)?;
            off = self.s.m.bin(c, Bin::Add, off, d);
        }
        let h = if a.param == CODE {
            None
        } else {
            let src = Source::Built { address: a, bits: 0, words: [0, 0] };
            Some(self.handle((b, i), &src, false, (11, 0))?)
        };
        let dword = |l: &mut Self, byte: V| -> Result<V, Why> {
            match h {
                None => l.code_dword(byte),
                Some(h) => {
                    let c = l.cur;
                    let (u, mask, al) = (l.s.m.undef(l.s.t.i32), l.s.m.i8(1), l.s.m.i32(4));
                    let r = l.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        l.s.t.res_ret,
                        139,
                        &[h, byte, u, mask, al],
                    );
                    let x = l.s.m.extract(c, r, 0);
                    Ok(l.as_int(x))
                }
            }
        };
        for k in 0..n {
            let x = match sub {
                None => {
                    let four = self.s.m.i32(4 * k);
                    let at = self.s.m.bin(c, Bin::Add, off, four);
                    dword(self, at)?
                }
                Some((width, signed)) => {
                    // shortcut: the field is taken not to cross a dword.
                    let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
                    let at = self.s.m.bin(c, Bin::And, off, not3);
                    let w = dword(self, at)?;
                    let byte = self.s.m.bin(c, Bin::And, off, three);
                    let sh = self.s.m.bin(c, Bin::Shl, byte, three);
                    let x = self.s.m.bin(c, Bin::LShr, w, sh);
                    let left = self.s.m.i32(32 - width);
                    let x = self.s.m.bin(c, Bin::Shl, x, left);
                    self.s.m.bin(c, if signed { Bin::AShr } else { Bin::LShr }, x, left)
                }
            };
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// `channels` channels of the element at byte `off` of raw buffer `h`,
    /// converted from the GFX10 buffer format `format` as the hardware's
    /// typed loads do (channels past the format's read 0, 0, 0, 1).
    // shortcut: the descriptor's channel swizzle is taken as the identity
    // (it is for raw views and the descriptors code builds), elements of 4
    // bytes or more as dword-aligned. Norm formats divide, which the AMD
    // driver does 1 ULP off the hardware's correctly rounded conversion for
    // some values; correct it with an integer check if a title compares
    // converted values exactly.
    fn format_load(&mut self, h: V, off: V, format: u8, channels: u32) -> Result<Vec<V>, Why> {
        let (bits, numeric) = buffer_format(format).ok_or("a buffer format")?;
        let c = self.cur;
        let total: u32 = bits.iter().sum();
        let (u, align) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
        let load = |l: &mut Self, at: V, n: u32| -> Vec<V> {
            let mask = l.s.m.i8((1u8 << n) - 1);
            let r = l.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                l.s.t.res_ret,
                139,
                &[h, at, u, mask, align],
            );
            (0..n)
                .map(|k| {
                    let x = l.s.m.extract(c, r, k);
                    l.as_int(x)
                })
                .collect()
        };
        let words = if total < 32 {
            // A byte or half inside the dword that holds it.
            let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
            let at = self.s.m.bin(c, Bin::And, off, not3);
            let w = load(self, at, 1)[0];
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            vec![self.s.m.bin(c, Bin::LShr, w, sh)]
        } else {
            load(self, off, total / 32)
        };
        let mut out = Vec::new();
        let mut at = 0;
        for &w in bits {
            let word = words[(at / 32) as usize];
            let field = if w == 32 {
                word
            } else {
                let sh = self.s.m.i32(at % 32);
                let t = self.s.m.bin(c, Bin::LShr, word, sh);
                let m = self.s.m.i32((1 << w) - 1);
                self.s.m.bin(c, Bin::And, t, m)
            };
            at += w;
            let x = self.convert(c, numeric, field, w);
            out.push(x);
        }
        let one = match numeric {
            Numeric::Uint | Numeric::Sint => 1,
            _ => 1.0f32.to_bits(),
        };
        for k in out.len() as u32..channels {
            out.push(self.s.m.i32(if k == 3 { one } else { 0 }));
        }
        out.truncate(channels as usize);
        Ok(out)
    }

    /// A typed load's channel `field`, `w` bits wide, converted as
    /// `numeric` says.
    fn convert(&mut self, c: usize, numeric: Numeric, field: V, w: u32) -> V {
        // Sign-extended from the field's top bit.
        let signed = |l: &mut Self| {
            if w == 32 {
                return field;
            }
            let k = l.s.m.i32(32 - w);
            let t = l.s.m.bin(c, Bin::Shl, field, k);
            l.s.m.bin(c, Bin::AShr, t, k)
        };
        let f32t = self.s.t.f32;
        match numeric {
            Numeric::Unorm => {
                let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                let max = self.s.m.f32((((1u64 << w) - 1) as f32).to_bits());
                let q = self.s.m.bin(c, Bin::SDiv, f, max);
                self.as_int(q)
            }
            Numeric::Snorm => {
                let s = signed(self);
                let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                let max = self.s.m.f32((((1u64 << (w - 1)) - 1) as f32).to_bits());
                let q = self.s.m.bin(c, Bin::SDiv, f, max);
                let lo = self.s.m.f32((-1.0f32).to_bits());
                let q = self.binary(35, q, lo);
                self.as_int(q)
            }
            Numeric::Uscaled => {
                let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                self.as_int(f)
            }
            Numeric::Sscaled => {
                let s = signed(self);
                let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                self.as_int(f)
            }
            Numeric::Uint => field,
            Numeric::Sint => signed(self),
            Numeric::Float if w == 32 => field,
            Numeric::Float => {
                let h = match half_shift(w) {
                    0 => field,
                    k => {
                        let sh = self.s.m.i32(k);
                        self.s.m.bin(c, Bin::Shl, field, sh)
                    }
                };
                let f = self.s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                self.as_int(f)
            }
        }
    }

    pub(super) fn buffer(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let a = self.buffer_access(inst, b, i)?;
        if a.used.is_some_and(|u| u.kind == ResourceType::Cbv) && !a.typed {
            return self.buffer_cbv(inst, &a);
        }
        let uav =
            a.store || a.atomic.is_some() || a.used.is_some_and(|u| u.kind == ResourceType::Uav);
        let u = if uav { 0x1000 } else { 0 };
        let iu = self.s.m.undef(self.s.t.i32);
        if let Some(op) = a.atomic {
            return self.buffer_atomic(inst, &a, op, (u, iu));
        }
        if a.store {
            return self.buffer_store(inst, &a, (u, iu));
        }
        if let Some(f) = a.format {
            return self.buffer_format_load(inst, &a, f, u);
        }
        let r = self.buffer_load(&a, uav, (u, iu))?;
        self.buffer_result(inst, &a, r)
    }

    /// The fields, view and byte offset of a buffer instruction.
    fn buffer_access(&mut self, inst: &Inst, b: usize, i: usize) -> Result<Access<'a>, Why> {
        let Fields::Buffer {
            data,
            addr,
            soffset,
            offset,
            offen,
            idxen,
            format,
            lds,
            tfe,
            cache,
            ..
        } = &inst.fields
        else {
            return Err("a buffer instruction without buffer fields");
        };
        let name = inst.name();
        if *lds || *tfe {
            return Err("a buffer access to LDS or with tfe");
        }
        if format.is_some() && !name.starts_with("tbuffer_load_format_") {
            return Err(name);
        }
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let atomic = atomic_op(name)?;
        // d16: two 16-bit channels per VGPR, low half first.
        let d16 = name.starts_with("buffer_") && name.contains("_format_d16_");
        let (typed, count) = shape(name, atomic.is_some()).ok_or(name)?;
        let Operand::Vector { reg: areg, .. } = *addr else {
            return Err("a buffer address");
        };
        let vaddr = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(areg + k)).ok_or("a buffer address")?;
            l.i32_of(v)
        };
        let index = if *idxen { Some(vaddr(self, 0)?) } else { None };
        let mut off = self.s.m.i32(u32::from(*offset));
        if *offen {
            let o = vaddr(self, u8::from(*idxen))?;
            off = self.s.m.bin(c, Bin::Add, off, o);
        }
        if !matches!(soffset, Operand::Scalar { code: NULL, .. } | Operand::Const(128)) {
            let so = self.operand(inst, soffset)?;
            off = self.s.m.bin(c, Bin::Add, off, so);
        }
        let src = self.descriptors.get(&(site, false)).copied().ok_or("an untracked buffer")?;
        let used = self.used_kind(src);
        let (base, extra) = self.built_offset(src)?;
        if base != 0 {
            let b = self.s.m.i32(base);
            off = self.s.m.bin(c, Bin::Add, off, b);
        }
        if let Some(x) = extra {
            off = self.s.m.bin(c, Bin::Add, off, x);
        }
        Ok(Access {
            site,
            guard,
            c,
            data: *data,
            offset: *offset,
            offen: *offen,
            format: *format,
            glc: cache.glc,
            store: name.starts_with("buffer_store_"),
            atomic,
            d16,
            typed,
            count,
            index,
            off,
            src,
            used,
        })
    }

    /// A structured buffer's handle, with the stride its used resource has.
    fn structured(&mut self, a: &Access, u: u32) -> Result<V, Why> {
        let stride = a.used.map_or(0, |u| u.stride);
        if stride < 4 || !stride.is_multiple_of(4) {
            return Err("a structured buffer of unknown stride");
        }
        self.handle(a.site, a.src, false, (12 | u, stride))
    }

    /// A constant buffer read per lane (a dynamically indexed array).
    // shortcut: the console's CBV descriptor is taken to have a 16-byte
    // stride, as the vector loads of whole rows suggest.
    fn buffer_cbv(&mut self, inst: &Inst, a: &Access) -> Result<(), Why> {
        let c = a.c;
        let h = self.handle(a.site, a.src, false, (13, 0x1_0000))?;
        let byte = match a.index {
            Some(i) => {
                let sixteen = self.s.m.i32(16);
                let rows = self.s.m.bin(c, Bin::Mul, i, sixteen);
                self.s.m.bin(c, Bin::Add, rows, a.off)
            }
            None => a.off,
        };
        let Some(Operand::Vector { reg, .. }) = a.data else {
            return Err("a buffer result");
        };
        for k in 0..a.count {
            let four = self.s.m.i32(4 * k);
            let at = self.s.m.bin(c, Bin::Add, byte, four);
            let x = self.cbuffer_dword(h, at);
            self.write(inst, Reg::V(reg + k as u8), x, a.guard);
        }
        Ok(())
    }

    /// DXIL atomic operation `op` on a raw or structured UAV; the old value
    /// returns with glc.
    fn buffer_atomic(
        &mut self,
        inst: &Inst,
        a: &Access,
        op: u32,
        (u, iu): (u32, V),
    ) -> Result<(), Why> {
        let d = a.data.as_ref().ok_or("buffer atomic data")?;
        let vals = self.vgprs(inst, d, if op == u32::MAX { 2 } else { 1 })?;
        let (h, coords) = match a.index {
            Some(index) => (self.structured(a, u)?, [index, a.off, iu]),
            None => (self.handle(a.site, a.src, false, (11 | u, 0))?, [a.off, iu, iu]),
        };
        let Operand::Vector { reg, .. } = *d else {
            return Err("buffer atomic data");
        };
        let old = match inst.read(Reg::V(reg)) {
            Some(v) if a.glc => self.i32_of(v)?,
            _ => self.s.m.i32(0),
        };
        let at = self.begin_guarded(a.guard);
        let i32t = self.s.t.i32;
        let r = if op == u32::MAX {
            // cmpswap: data is (new, compare).
            let mut args = vec![h];
            args.extend(coords);
            args.extend([vals[1], vals[0]]);
            self.s.op(self.cur, "atomicCompareExchange.i32", Attr::NoUnwind, i32t, 79, &args)
        } else {
            let mut args = vec![h, self.s.m.i32(op)];
            args.extend(coords);
            args.push(vals[0]);
            self.s.op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &args)
        };
        let r = self.end_guarded(at, &[(r, old)])[0];
        if a.glc {
            self.write(inst, Reg::V(reg), r, None);
        }
        Ok(())
    }

    fn buffer_store(&mut self, inst: &Inst, a: &Access, (u, iu): (u32, V)) -> Result<(), Why> {
        let c = a.c;
        let regs = if a.d16 { a.count.div_ceil(2) } else { a.count };
        let mut vals = self.vgprs(inst, a.data.as_ref().ok_or("buffer store data")?, regs as u8)?;
        if a.d16 {
            // shortcut: d16 data is taken as f16 (integer formats would
            // keep 16 bits), widened for the 32-bit float store.
            let mut wide = Vec::new();
            for k in 0..a.count {
                let mut x = vals[(k / 2) as usize];
                if k % 2 == 1 {
                    let k16 = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, k16);
                }
                let f32t = self.s.t.f32;
                let f = self.s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                wide.push(self.as_int(f));
            }
            vals = wide;
        }
        let mut args = Vec::new();
        let (op, name) = if a.typed {
            // shortcut: typed stores are taken to write all four channels
            // of a float format.
            if a.count != 4 || a.offen || a.offset != 0 {
                return Err("a typed buffer store of some channels or at an offset");
            }
            args.push(self.handle(a.site, a.src, false, (10 | u, 0x409))?);
            args.extend([a.index.unwrap_or(self.s.m.i32(0)), iu]);
            for x in vals {
                args.push(self.as_float(x));
            }
            (69, "bufferStore.f32")
        } else {
            match a.index {
                Some(index) => {
                    args.push(self.structured(a, u)?);
                    args.extend([index, a.off]);
                }
                None => {
                    args.push(self.handle(a.site, a.src, false, (11 | u, 0))?);
                    args.extend([a.off, iu]);
                }
            }
            args.extend(&vals);
            args.resize(3 + 4, iu);
            (140, "rawBufferStore.i32")
        };
        args.push(self.s.m.i8(if a.typed { 0xF } else { (1u8 << a.count) - 1 }));
        if !a.typed {
            args.push(self.s.m.i32(4));
        }
        let at = self.begin_guarded(a.guard);
        let void = self.s.t.void;
        self.s.op(self.cur, name, Attr::NoUnwind, void, op, &args);
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// A `tbuffer_load_format_*`: channels in the instruction's own format.
    fn buffer_format_load(&mut self, inst: &Inst, a: &Access, f: u8, u: u32) -> Result<(), Why> {
        if a.index.is_some() {
            return Err("a typed-format load by index");
        }
        let h = self.handle(a.site, a.src, false, (11 | u, 0))?;
        let Some(Operand::Vector { reg, .. }) = a.data else {
            return Err("a buffer result");
        };
        for (k, x) in self.format_load(h, a.off, f, a.count)?.into_iter().enumerate() {
            self.write(inst, Reg::V(reg + k as u8), x, a.guard);
        }
        Ok(())
    }

    /// A typed load (in the view's format), or a raw or structured one.
    fn buffer_load(&mut self, a: &Access, uav: bool, (u, iu): (u32, V)) -> Result<V, Why> {
        let c = a.c;
        if a.typed {
            if a.offen || matches!(a.off, V::Inst(_)) || a.offset != 0 {
                return Err("a typed buffer load at an offset");
            }
            if uav {
                self.s.flags |= flags::TYPED_UAV_LOADS;
            }
            let h = self.handle(a.site, a.src, false, (10 | u, 0x409))?;
            let index = a.index.unwrap_or(self.s.m.i32(0));
            let ret = self.s.t.res_ret;
            return Ok(self.s.op(c, "bufferLoad.f32", Attr::ReadOnly, ret, 68, &[h, index, iu]));
        }
        let mask = self.s.m.i8((1u8 << a.count) - 1);
        let align = self.s.m.i32(4);
        let (h, [x, y]) = match a.index {
            Some(index) => (self.structured(a, u)?, [index, a.off]),
            None => (self.handle(a.site, a.src, false, (11 | u, 0))?, [a.off, iu]),
        };
        let args = [h, x, y, mask, align];
        Ok(self.s.op(c, "rawBufferLoad.f32", Attr::ReadOnly, self.s.t.res_ret, 139, &args))
    }

    /// Writes the dwords or channels of a load result `r`.
    fn buffer_result(&mut self, inst: &Inst, a: &Access, r: V) -> Result<(), Why> {
        let c = a.c;
        let Some(Operand::Vector { reg, .. }) = a.data else {
            return Err("a buffer result");
        };
        if a.d16 {
            // Two f16 channels per VGPR (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            let i32t = self.s.t.i32;
            let f16 = |l: &mut Self, ch| {
                let f = l.s.m.extract(c, r, ch);
                l.s.op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
            };
            for k in 0..a.count.div_ceil(2) {
                // Channel 2k + 1 (the high half), if any, is converted first.
                let lo = 2 * k;
                let hi = (lo + 1 < a.count).then(|| f16(self, lo + 1));
                let h = f16(self, lo);
                let x = match hi {
                    Some(hi) => {
                        let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                        self.s.m.bin(c, Bin::Or, h, hi)
                    }
                    None => h,
                };
                self.write(inst, Reg::V(reg + k as u8), x, a.guard);
            }
            return Ok(());
        }
        for k in 0..a.count {
            let x = self.s.m.extract(c, r, k);
            let x = self.as_int(x);
            self.write(inst, Reg::V(reg + k as u8), x, a.guard);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_formats_by_number() {
        // The numbers the disassembler names 8_8_8_8_UNORM, 16_16_SNORM,
        // 2_10_10_10_UNORM, 16_FLOAT, 32_32_32_32_FLOAT.
        assert_eq!(buffer_format(56), Some((&[8, 8, 8, 8][..], Numeric::Unorm)));
        assert_eq!(buffer_format(24), Some((&[16, 16][..], Numeric::Snorm)));
        assert_eq!(buffer_format(50), Some((&[10, 10, 10, 2][..], Numeric::Unorm)));
        assert_eq!(buffer_format(13), Some((&[16][..], Numeric::Float)));
        assert_eq!(buffer_format(77), Some((&[32, 32, 32, 32][..], Numeric::Float)));
        // Invalid, past the table.
        assert_eq!(buffer_format(0), None);
        assert_eq!(buffer_format(78), None);
    }

    #[test]
    fn packed_buffer_formats_by_number() {
        // LLVM's `UfmtGFX10`: 10_11_11 is 30-36, 11_11_10 37-43,
        // 10_10_10_2 44-49. Channels from the lowest bits: R11G11B10_FLOAT
        // is 10_11_11 (Mesa), 10_10_10_2 has the 2-bit channel first.
        assert_eq!(buffer_format(30), Some((&[11, 11, 10][..], Numeric::Unorm)));
        assert_eq!(buffer_format(36), Some((&[11, 11, 10][..], Numeric::Float)));
        assert_eq!(buffer_format(37), Some((&[10, 11, 11][..], Numeric::Unorm)));
        assert_eq!(buffer_format(43), Some((&[10, 11, 11][..], Numeric::Float)));
        assert_eq!(buffer_format(44), Some((&[2, 10, 10, 10][..], Numeric::Unorm)));
        assert_eq!(buffer_format(45), Some((&[2, 10, 10, 10][..], Numeric::Snorm)));
        assert_eq!(buffer_format(46), Some((&[2, 10, 10, 10][..], Numeric::Uscaled)));
        assert_eq!(buffer_format(47), Some((&[2, 10, 10, 10][..], Numeric::Sscaled)));
        assert_eq!(buffer_format(48), Some((&[2, 10, 10, 10][..], Numeric::Uint)));
        assert_eq!(buffer_format(49), Some((&[2, 10, 10, 10][..], Numeric::Sint)));
    }

    /// A half's value (sign, 5-bit exponent biased 15, 10-bit mantissa), as
    /// `legacyF16ToF32` widens it.
    fn half(h: u32) -> f32 {
        let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 31, h & 1023);
        let v = match e {
            0 => m as f32 * 2f32.powi(-24),
            31 if m == 0 => f32::INFINITY,
            31 => f32::NAN,
            _ => (1024 + m) as f32 * 2f32.powi(e as i32 - 25),
        };
        if s == 1 { -v } else { v }
    }

    #[test]
    fn small_floats_widen_as_halves() {
        // Unsigned 11- and 10-bit floats: a 5-bit exponent biased 15 over a
        // 6- or 5-bit mantissa, no sign; values from the D3D/GL spec.
        let f11 = |x: u32| half(x << half_shift(11));
        let f10 = |x: u32| half(x << half_shift(10));
        assert_eq!(f11(0), 0.0);
        assert_eq!(f11(0x001), 2f32.powi(-20)); // smallest denormal
        assert_eq!(f11(0x03F), 63.0 * 2f32.powi(-20)); // largest denormal
        assert_eq!(f11(0x040), 2f32.powi(-14)); // smallest normal
        assert_eq!(f11(0x3C0), 1.0);
        assert_eq!(f11(0x3E0), 1.5);
        assert_eq!(f11(0x7BF), 65024.0); // largest
        assert_eq!(f11(0x7C0), f32::INFINITY);
        assert!(f11(0x7C1).is_nan());
        assert_eq!(f10(0x001), 2f32.powi(-19));
        assert_eq!(f10(0x020), 2f32.powi(-14));
        assert_eq!(f10(0x1E0), 1.0);
        assert_eq!(f10(0x3DF), 64512.0);
        assert_eq!(f10(0x3E0), f32::INFINITY);
        assert!(f10(0x3FF).is_nan());
        // A 16-bit float is a half already, sign included.
        assert_eq!(half(0xBC00 << half_shift(16)), -1.0);
    }
}

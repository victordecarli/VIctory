//! Image instructions: loads, stores, atomics, sampling, gathers and size
//! queries, through the texture and sampler handles `resource` traced.

use super::*;

/// What the parts of one image instruction share.
struct Access<'a> {
    name: &'static str,
    site: Site,
    guard: Option<V>,
    /// The block the instruction starts in.
    c: usize,
    data: Operand,
    /// The channels, at least one.
    dmask: u8,
    /// DXIL's resource kind, and how many coordinates address it.
    kind: u32,
    coords: usize,
    /// The address VGPRs in order.
    regs: Vec<u8>,
    unorm: bool,
    a16: bool,
    d16: bool,
    glc: bool,
    src: &'a Source,
    store: bool,
    uav: bool,
    /// The texture handle's properties.
    props: (u32, u32),
}

/// Where a sample's addresses are, past its coordinates.
struct Sampled {
    /// The first address after the offset word, bias and reference.
    first: usize,
    /// Coordinates the size divides (not the array slice).
    spatial: usize,
    /// Texel offsets (`_o`) come first.
    offset_word: bool,
    /// Offset components DXIL takes.
    offsets: usize,
    fu: V,
}

impl<'a> Lower<'a> {
    pub(super) fn image(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let a = self.image_access(inst, b, i)?;
        if let Some(op) = a.name.strip_prefix("image_atomic_") {
            return self.image_atomic(inst, &a, op);
        }
        if a.store {
            return self.image_store(inst, &a);
        }
        let (op, sample) = match a.name {
            "image_sample" | "image_sample_o" => (60, true),
            "image_sample_lz" | "image_sample_l" => (62, true),
            "image_sample_b" => (61, true),
            "image_sample_c_lz" => (65, true),
            "image_sample_d" => (63, true),
            "image_get_lod" if self.s.pixel => (81, true),
            // Implicit derivatives: pixel shaders only.
            "image_sample_c" if self.s.pixel => (64, true),
            "image_gather4" | "image_gather4_lz" | "image_gather4_lz_o" => (73, true),
            "image_load" | "image_load_mip" => (66, false),
            "image_get_resinfo" => (72, false),
            _ => return Err(a.name),
        };
        if op == 72 {
            return self.image_size(inst, &a);
        }
        if a.uav && op != 66 {
            return Err("sampling a UAV");
        }
        let h = self.handle(a.site, a.src, false, a.props)?;
        let fu = self.s.m.undef(self.s.t.f32);
        let iu = self.s.m.undef(self.s.t.i32);
        let r = if sample {
            // A level-of-detail query writes its own result.
            let Some(r) = self.image_sample(inst, &a, op, h, (fu, iu))? else {
                return Ok(());
            };
            r
        } else {
            self.image_load(inst, &a, h, iu)?
        };
        self.image_result(inst, &a, op, r)
    }

    /// The fields, address registers and view of an image instruction.
    fn image_access(&mut self, inst: &Inst, b: usize, i: usize) -> Result<Access<'a>, Why> {
        let Fields::Image { data, addr, dmask, dim, unorm, a16, d16, tfe, lwe, cache, .. } =
            &inst.fields
        else {
            return Err("an image instruction without image fields");
        };
        if *tfe || *lwe {
            return Err("an image access with tfe or lwe");
        }
        let name = inst.name();
        if *a16 && name.starts_with("image_sample_b") {
            return Err("a biased image sample with 16-bit coordinates");
        }
        // A load or sample with no channel enabled returns the first.
        let dmask = if *dmask == 0 { 1 } else { *dmask };
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let (kind, coords) = match dim {
            0 => (1, 1),
            1 => (2, 2),
            2 => (4, 3),
            // Sampled (face coordinates and face, see `cube_direction`) or
            // sized only.
            3 if name.starts_with("image_sample")
                || name.starts_with("image_gather")
                || name == "image_get_resinfo" =>
            {
                (5, 3)
            }
            4 => (6, 2),
            5 => (7, 3),
            _ => return Err("an image dimension (cube loads or MSAA)"),
        };
        let mut regs = Vec::new();
        for a in addr {
            if let Operand::Vector { reg, dwords } = *a {
                regs.extend((0..dwords).map(|k| reg + k));
            }
        }
        let src = self.descriptors.get(&(site, false)).copied().ok_or("an untracked image")?;
        let store = name.starts_with("image_store");
        let uav = store || self.used_kind(src).is_some_and(|u| u.kind == ResourceType::Uav);
        let props = (kind | if uav { 0x1000 } else { 0 }, 0x409);
        Ok(Access {
            name,
            site,
            guard,
            c,
            data: *data,
            dmask,
            kind,
            coords,
            regs,
            unorm: *unorm,
            a16: *a16,
            d16: *d16,
            glc: cache.glc,
            src,
            store,
            uav,
            props,
        })
    }

    /// Address `k`. With a16, two 16-bit components per VGPR, low half first.
    fn image_addr(&mut self, inst: &Inst, a: &Access, k: usize) -> Result<V, Why> {
        let r = *a.regs.get(if a.a16 { k / 2 } else { k }).ok_or("an image address")?;
        let v = inst.read(Reg::V(r)).ok_or("an image address")?;
        let x = self.i32_of(v)?;
        if !a.a16 {
            return Ok(x);
        }
        let c = self.cur;
        let (sh, mask) = (self.s.m.i32(16 * (k as u32 % 2)), self.s.m.i32(0xFFFF));
        let x = self.s.m.bin(c, Bin::LShr, x, sh);
        Ok(self.s.m.bin(c, Bin::And, x, mask))
    }

    /// A sampling coordinate or level: with a16 they are f16.
    fn image_float(&mut self, a: &Access, x: V) -> V {
        if a.a16 {
            let f = self.s.t.f32;
            self.s.op(self.cur, "legacyF16ToF32", Attr::ReadNone, f, 131, &[x])
        } else {
            self.as_float(x)
        }
    }

    /// The first `coords` addresses, as integers.
    fn image_coords(&mut self, inst: &Inst, a: &Access) -> Result<Vec<V>, Why> {
        let iu = self.s.m.undef(self.s.t.i32);
        let mut cs = vec![iu; 3];
        for (k, slot) in cs.iter_mut().enumerate().take(a.coords) {
            *slot = self.image_addr(inst, a, k)?;
        }
        Ok(cs)
    }

    /// DXIL's atomic operations on a 32-bit integer texture UAV; the old
    /// value returns with glc. Sub is the add of the negation.
    fn image_atomic(&mut self, inst: &Inst, a: &Access, op: &str) -> Result<(), Why> {
        let code = match op {
            "add" | "sub" => 0,
            "and" => 1,
            "or" => 2,
            "xor" => 3,
            "smin" => 4,
            "smax" => 5,
            "umin" => 6,
            "umax" => 7,
            "swap" => 8,
            "cmpswap" => u32::MAX,
            _ => return Err(a.name),
        };
        if a.d16 || a.a16 {
            return Err("a 16-bit image atomic");
        }
        // dmask: the data dwords (cmpswap: twice the value's).
        // shortcut: 64-bit image atomics (two dwords, an R32G32 view)
        // need the host's 64-bit typed atomics; they stop the program.
        if a.dmask != if code == u32::MAX { 3 } else { 1 } {
            return Err("a 64-bit image atomic");
        }
        let cs = self.image_coords(inst, a)?;
        let Operand::Vector { reg, .. } = a.data else {
            return Err("image atomic data");
        };
        let vals = self.vgprs(inst, &a.data, if code == u32::MAX { 2 } else { 1 })?;
        let mut value = vals[0];
        if op == "sub" {
            let z = self.s.m.i32(0);
            value = self.s.m.bin(a.c, Bin::Sub, z, value);
        }
        let old = match inst.read(Reg::V(reg)) {
            Some(v) if a.glc => self.i32_of(v)?,
            _ => self.s.m.i32(0),
        };
        let h = self.handle(a.site, a.src, false, (a.kind | 0x1000, 0x105))?;
        let at = self.begin_guarded(a.guard);
        let i32t = self.s.t.i32;
        let mut args = vec![h];
        let r = if code == u32::MAX {
            // cmpswap: data is (new, compare).
            args.extend(cs);
            args.extend([vals[1], vals[0]]);
            self.s.op(self.cur, "atomicCompareExchange.i32", Attr::NoUnwind, i32t, 79, &args)
        } else {
            args.push(self.s.m.i32(code));
            args.extend(cs);
            args.push(value);
            self.s.op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &args)
        };
        let r = self.end_guarded(at, &[(r, old)])[0];
        if a.glc {
            self.write(inst, Reg::V(reg), r, None);
        }
        Ok(())
    }

    fn image_store(&mut self, inst: &Inst, a: &Access) -> Result<(), Why> {
        if a.name != "image_store" {
            return Err("an image store to a mip level");
        }
        // shortcut: stores are taken to write all four channels of a
        // float format (the console converts by the view's format).
        if a.dmask != 0xF {
            return Err("an image store of some channels");
        }
        let cs = self.image_coords(inst, a)?;
        let Operand::Vector { reg, .. } = a.data else {
            return Err("image store data");
        };
        let mut vals = Vec::new();
        for k in 0..4u8 {
            // d16: two f16 channels per VGPR, low half first.
            let r = if a.d16 { k / 2 } else { k };
            let v = inst.read(Reg::V(reg + r)).ok_or("image store data")?;
            vals.push(if a.d16 {
                let mut x = self.i32_of(v)?;
                if k % 2 == 1 {
                    let k16 = self.s.m.i32(16);
                    x = self.s.m.bin(self.cur, Bin::LShr, x, k16);
                }
                let f32t = self.s.t.f32;
                self.s.op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
            } else {
                self.float_of(v)?
            });
        }
        let h = self.handle(a.site, a.src, false, a.props)?;
        let at = self.begin_guarded(a.guard);
        let mut args = vec![h];
        args.extend(cs);
        args.extend(vals);
        args.push(self.s.m.i8(0xF));
        let void = self.s.t.void;
        self.s.op(self.cur, "textureStore.f32", Attr::NoUnwind, void, 67, &args);
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Width, height, depth or layers, levels (per dmask) of the level in
    /// the first address: DXIL's dimensions; a channel DXIL leaves out reads
    /// 1 (UAVs: one level).
    fn image_size(&mut self, inst: &Inst, a: &Access) -> Result<(), Why> {
        let c = a.c;
        let h = self.handle(a.site, a.src, false, a.props)?;
        let mip = if a.uav { self.s.m.undef(self.s.t.i32) } else { self.image_addr(inst, a, 0)? };
        let i32t = self.s.t.i32;
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let d = self.s.op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
        let one = self.s.m.i32(1);
        let present = match a.kind {
            1 => [true, false, false, true],
            2 | 5 | 6 => [true, true, false, true],
            _ => [true, true, true, true],
        };
        let Operand::Vector { reg, .. } = a.data else {
            return Err("an image result");
        };
        for (k, ch) in (0..4u32).filter(|ch| a.dmask >> ch & 1 != 0).enumerate() {
            let x = if ch == 3 && a.uav || !present[ch as usize] {
                one
            } else {
                self.s.m.extract(c, d, ch)
            };
            self.write(inst, Reg::V(reg + k as u8), x, a.guard);
        }
        Ok(())
    }

    /// A sample or gather of texture `h`: its result, or `None` for a
    /// level-of-detail query, which writes its own.
    fn image_sample(
        &mut self,
        inst: &Inst,
        a: &Access,
        op: u32,
        h: V,
        (fu, iu): (V, V),
    ) -> Result<Option<V>, Why> {
        let (c, name) = (a.c, a.name);
        let ss = self.descriptors.get(&(a.site, true)).copied().ok_or("an untracked sampler")?;
        // Comparison: the reference value is the first address after
        // the bias, and the sampler is a comparison sampler.
        let cmp = matches!(op, 64 | 65);
        if cmp && (a.a16 || a.dmask.count_ones() != 1) {
            return Err("a comparison sample of 16-bit coordinates or several channels");
        }
        let sh = self.handle(a.site, ss, true, (14 | if cmp { 0x8000 } else { 0 }, 0))?;
        // Texel offsets (`_o`) come first: x in bits 0-5, y in 8-13,
        // signed (programmable offsets: gathers only).
        let offset_word = name.ends_with("_o");
        let first =
            usize::from(offset_word) + usize::from(name == "image_sample_b") + usize::from(cmp);
        // Coordinates the size divides (not the array slice); explicit
        // gradients (x then y) come before the coordinates.
        let spatial = a.coords - usize::from(a.kind == 6 || a.kind == 7);
        let grads = if op == 63 { 2 * spatial } else { 0 };
        let mut cs = vec![fu; 4];
        for (k, slot) in cs.iter_mut().enumerate().take(a.coords) {
            let x = self.image_addr(inst, a, first + grads + k)?;
            *slot = self.image_float(a, x);
        }
        if a.kind == 5 {
            if a.unorm || op == 63 || op == 81 {
                return Err("a cube sample with texel coordinates, gradients or a level query");
            }
            let dir = self.cube_direction(cs[0], cs[1], cs[2]);
            cs[..3].copy_from_slice(&dir);
        }
        if a.unorm {
            // Texel coordinates: divided by the size of level 0.
            let i32t = self.s.t.i32;
            let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
            let z = self.s.m.i32(0);
            let d = self.s.op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, z]);
            for (k, slot) in cs.iter_mut().enumerate().take(spatial) {
                let n = self.s.m.extract(c, d, k as u32);
                let n = self.s.m.cast(c, Cast::UiToFp, n, self.s.t.f32);
                *slot = self.s.m.bin(c, Bin::SDiv, *slot, n);
            }
        }
        if op == 81 {
            self.image_lod(inst, a, [h, sh], &cs)?;
            return Ok(None);
        }
        let offsets = if a.kind == 5 {
            0
        } else {
            (a.coords - usize::from(a.kind == 6 || a.kind == 7)).min(3)
        };
        let zero = self.s.m.i32(0);
        let mut os = vec![iu; 3];
        for o in os.iter_mut().take(offsets) {
            *o = zero;
        }
        let mut args = vec![h, sh];
        args.extend(&cs);
        let s = Sampled { first, spatial, offset_word, offsets, fu };
        self.sample_op(inst, a, op, &s, (os, args)).map(Some)
    }

    /// The clamped then the unclamped level of detail.
    fn image_lod(&mut self, inst: &Inst, a: &Access, [h, sh]: [V; 2], cs: &[V]) -> Result<(), Why> {
        let c = a.c;
        let Operand::Vector { reg, .. } = a.data else {
            return Err("an image result");
        };
        let f32t = self.s.t.f32;
        for (k, ch) in (0..2u32).filter(|ch| a.dmask >> ch & 1 != 0).enumerate() {
            let clamped = self.s.m.bool(ch == 0);
            let lod = self.s.op(
                c,
                "calculateLOD.f32",
                Attr::ReadOnly,
                f32t,
                81,
                &[h, sh, cs[0], cs[1], cs[2], clamped],
            );
            let x = self.as_int(lod);
            self.write(inst, Reg::V(reg + k as u8), x, a.guard);
        }
        Ok(())
    }

    /// The DXIL sample or gather `op`, on the handles and coordinates in
    /// `args`, with the offsets `os`.
    fn sample_op(
        &mut self,
        inst: &Inst,
        a: &Access,
        op: u32,
        s: &Sampled,
        (mut os, mut args): (Vec<V>, Vec<V>),
    ) -> Result<V, Why> {
        let (c, fu, ret) = (a.c, s.fu, self.s.t.res_ret);
        match op {
            60 => {
                if s.offset_word {
                    self.sample_offsets(inst, a, s.offsets, &mut os)?;
                }
                args.extend(&os);
                args.push(fu);
                Ok(self.s.op(c, "sample.f32", Attr::ReadOnly, ret, 60, &args))
            }
            61 => {
                let bias = self.image_addr(inst, a, 0)?;
                let bias = self.as_float(bias);
                args.extend(&os);
                args.extend([bias, fu]);
                Ok(self.s.op(c, "sampleBias.f32", Attr::ReadOnly, ret, 61, &args))
            }
            63 => {
                let mut g = vec![fu; 6];
                for k in 0..s.spatial {
                    let x = self.image_addr(inst, a, s.first + k)?;
                    g[k] = self.image_float(a, x);
                    let y = self.image_addr(inst, a, s.first + s.spatial + k)?;
                    g[3 + k] = self.image_float(a, y);
                }
                args.extend(&os);
                args.extend(g);
                args.push(fu);
                Ok(self.s.op(c, "sampleGrad.f32", Attr::ReadOnly, ret, 63, &args))
            }
            64 | 65 => {
                let reference = self.image_addr(inst, a, s.first - 1)?;
                let reference = self.as_float(reference);
                args.extend(&os);
                args.push(reference);
                let name = if op == 64 {
                    args.push(fu);
                    "sampleCmp.f32"
                } else {
                    "sampleCmpLevelZero.f32"
                };
                Ok(self.s.op(c, name, Attr::ReadOnly, ret, op, &args))
            }
            62 => {
                let lod = if a.name == "image_sample_l" {
                    let x = self.image_addr(inst, a, a.coords)?;
                    self.image_float(a, x)
                } else {
                    self.s.m.f32(0)
                };
                args.extend(&os);
                args.push(lod);
                Ok(self.s.op(c, "sampleLevel.f32", Attr::ReadOnly, ret, 62, &args))
            }
            _ => self.gather(inst, a, s, (os, args)),
        }
    }

    /// A sample's offsets from its offset word. DXIL takes immediate
    /// offsets in -8..=7; the console's are 6-bit and may be computed.
    fn sample_offsets(
        &mut self,
        inst: &Inst,
        a: &Access,
        offsets: usize,
        os: &mut [V],
    ) -> Result<(), Why> {
        let w = self.image_addr(inst, a, 0)?;
        let w = self.s.m.int_value(w).ok_or("a sample offset that is not constant")? as u32;
        for (k, o) in os.iter_mut().enumerate().take(offsets) {
            let x = (w >> (8 * k) & 0x3F) as i32;
            let x = if x >= 32 { x - 64 } else { x };
            if !(-8..=7).contains(&x) {
                return Err("a sample offset past -8..7");
            }
            *o = self.s.m.i32(x as u32);
        }
        Ok(())
    }

    /// A gather of one channel at level 0, with programmable offsets.
    fn gather(
        &mut self,
        inst: &Inst,
        a: &Access,
        s: &Sampled,
        (mut os, mut args): (Vec<V>, Vec<V>),
    ) -> Result<V, Why> {
        let c = a.c;
        if a.name == "image_gather4" {
            return Err("a gather with implicit level");
        }
        if a.dmask.count_ones() != 1 {
            return Err("a gather of several channels");
        }
        if s.offset_word {
            let w = self.image_addr(inst, a, 0)?;
            for (k, at) in [(0, 26), (1, 18)] {
                // Bits at..at+5 sign-extended: shift up, then down.
                let left = self.s.m.i32(at);
                let t = self.s.m.bin(c, Bin::Shl, w, left);
                let k26 = self.s.m.i32(26);
                os[k] = self.s.m.bin(c, Bin::AShr, t, k26);
            }
        }
        args.extend(&os[..2]);
        args.push(self.s.m.i32(a.dmask.trailing_zeros()));
        Ok(self.s.op(c, "textureGather.f32", Attr::ReadOnly, self.s.t.res_ret, 73, &args))
    }

    fn image_load(&mut self, inst: &Inst, a: &Access, h: V, iu: V) -> Result<V, Why> {
        let cs = self.image_coords(inst, a)?;
        let mip = if a.name == "image_load_mip" {
            self.image_addr(inst, a, a.coords)?
        } else if a.uav {
            // Our declarations have four channels.
            self.s.flags |= flags::TYPED_UAV_LOADS;
            iu
        } else {
            self.s.m.i32(0)
        };
        let mut args = vec![h, mip];
        args.extend(&cs);
        args.extend([iu, iu, iu]);
        Ok(self.s.op(a.c, "textureLoad.f32", Attr::ReadOnly, self.s.t.res_ret, 66, &args))
    }

    /// Writes the channels of a load, sample or gather result `r`.
    fn image_result(&mut self, inst: &Inst, a: &Access, op: u32, r: V) -> Result<(), Why> {
        let c = a.c;
        let Operand::Vector { reg, .. } = a.data else {
            return Err("an image result");
        };
        let channels: Vec<u32> = if op == 73 {
            vec![0, 1, 2, 3]
        } else {
            (0..4).filter(|ch| a.dmask >> ch & 1 != 0).collect()
        };
        let mut xs = Vec::new();
        for ch in channels {
            let x = self.s.m.extract(c, r, ch);
            xs.push(if a.d16 {
                // shortcut: d16 results are taken as floats (rounded to
                // nearest even); integer formats would keep 16 bits.
                let i32t = self.s.t.i32;
                self.s.op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[x])
            } else {
                self.as_int(x)
            });
        }
        if a.d16 {
            // Two channels per VGPR, low half first (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            xs = xs
                .chunks(2)
                .map(|p| match *p {
                    [lo, hi] => {
                        let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                        self.s.m.bin(c, Bin::Or, lo, hi)
                    }
                    _ => p[0],
                })
                .collect();
        }
        for (k, x) in xs.into_iter().enumerate() {
            self.write(inst, Reg::V(reg + k as u8), x, a.guard);
        }
        Ok(())
    }

    /// The direction a cube sample's face coordinates stand for: the
    /// console takes s and t in [1, 2] (`v_cubesc`/`v_cubetc` over
    /// `|v_cubema|`, plus 1.5) and the face id (`v_cubeid`), which the
    /// host's cube sampling turns back into the same texel.
    // shortcut: the view is taken as a cube, not a cube array (whose face
    // operand adds 8 × the slice).
    fn cube_direction(&mut self, s: V, t: V, face: V) -> [V; 3] {
        let c = self.cur;
        let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
        let (two, three) = (k(self, 2.0), k(self, 3.0));
        let u = self.s.m.bin(c, Bin::Mul, s, two);
        let u = self.s.m.bin(c, Bin::Sub, u, three);
        let v = self.s.m.bin(c, Bin::Mul, t, two);
        let v = self.s.m.bin(c, Bin::Sub, v, three);
        let z = self.s.m.f32(0x8000_0000);
        let (nu, nv) = (self.s.m.bin(c, Bin::Sub, z, u), self.s.m.bin(c, Bin::Sub, z, v));
        let (one, mone) = (k(self, 1.0), k(self, -1.0));
        let f = self.s.m.cast(c, Cast::FpToUi, face, self.s.t.i32);
        // Per face +x, -x, +y, -y, +z, -z.
        let table =
            [[one, mone, u, u, u, nu], [nv, nv, one, mone, nv, nv], [nu, u, v, nv, one, mone]];
        table.map(|row| {
            let mut r = row[0];
            for (n, &x) in row.iter().enumerate().skip(1) {
                let kn = self.s.m.i32(n as u32);
                let is = self.s.m.cmp(c, pred::EQ, f, kn);
                r = self.s.m.select(c, is, x, r);
            }
            r
        })
    }
}

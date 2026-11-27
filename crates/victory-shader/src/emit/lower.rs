//! IR to DXIL, one machine instruction at a time.
//!
//! Every register dword is an `i32` (SCC an `i1`); float operations bitcast
//! their operands. Lane masks exist only as the per-lane booleans of
//! structurisation; branches and guards are built from them. A value the
//! emitter cannot express is `Missing` with a reason, and only fails the
//! program when something with an effect (an export, a branch) needs it.

use std::collections::{BTreeMap, HashMap, HashSet};

use super::UserData;
use super::bitcode::{Attr, Bin, Cast, Term, V, pred};
use super::container::SigElement;
use super::shader::{Shader, flags};
use crate::ShaderError;
use crate::gfx10::{Dpp, Fields, NULL, Operand};
use crate::io::{Barycentric, Export, Input, Interp, Io, Target};
use crate::ir::{EXEC, Exit, Inst, Program, Reg, Value};
use crate::native::{HwStage, NativeProgram, ResourceType, RootType, StageIo, Usage, UsedResource};
use crate::resource::{Address, CODE, Resources, Role, Site, Source};
use crate::structure::{Atom, Bool, LaneExit, Structured};

mod control;
mod descriptor;
mod halves;
mod image;
mod integer;
mod io;
mod lds;
mod memory;
mod salu;
mod setup;
mod smem;
mod valu;
mod valu16;

type Why = &'static str;

#[derive(Debug, Clone, Copy)]
enum Val {
    I32(V),
    Bit(V),
    Missing(Why),
}

pub(crate) struct Lowered {
    pub dxil: Vec<u8>,
    /// Bit n: immediate sampler n is used for comparisons.
    pub cmp_samplers: u64,
    pub root_cbvs: Vec<u8>,
    pub root_srvs: Vec<u8>,
    pub user_data: Vec<UserData>,
    pub samplers: Vec<[u32; 4]>,
    /// The lanes per wave the program declares it needs (`WaveSize`).
    pub wave_size: Option<u32>,
    /// Heap handles made with a non-uniform index.
    #[cfg(test)]
    pub nonuniform_handles: u32,
    /// Whether the program uses wave operations.
    #[cfg(test)]
    pub wave_ops: bool,
    /// Group barriers emitted.
    #[cfg(test)]
    pub barriers: u32,
    /// Reads of another lane emitted.
    #[cfg(test)]
    pub lane_reads: u32,
    /// Memory accesses run only by a console wave that would enter their
    /// group-wide branch ([`halves`]).
    #[cfg(test)]
    pub flagged_accesses: u32,
    /// Votes of the whole group at a branch it takes as one ([`halves`]).
    #[cfg(test)]
    pub group_votes: u32,
}

pub(crate) struct Parts<'a> {
    pub native: &'a NativeProgram,
    pub p: &'a Program,
    pub st: &'a Structured,
    pub io: &'a Io,
    pub res: &'a Resources,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    pub prims: Vec<SigElement>,
    /// Mesh: what the group id arrives multiplied by.
    pub mesh_stride: u32,
    /// The most lanes a host wave has (see [`super::translate`]).
    pub host_lanes: u32,
}

enum Stop {
    /// A phi whose incoming value has no DXIL form was read: redo with it missing.
    Retry(Value, Why),
    Fail(ShaderError),
}

pub(crate) fn lower(parts: &Parts) -> Result<Lowered, ShaderError> {
    let first = lower_once(parts)?;
    Ok(halves::widened(parts, first))
}

/// [`lower`], with the lane branches `parts.st` has.
fn lower_once(parts: &Parts) -> Result<Lowered, ShaderError> {
    let mut bad = HashMap::new();
    loop {
        match Lower::new(parts, &bad).run() {
            Ok(l) => return Ok(l),
            Err(Stop::Retry(v, why)) => {
                if bad.insert(v, why).is_some() {
                    return Err(ShaderError::Unsupported {
                        offset: 0,
                        what: "a phi that does not settle",
                    });
                }
            }
            Err(Stop::Fail(e)) => return Err(e),
        }
    }
}

const TWO_PI: f32 = std::f32::consts::TAU;

struct Lower<'a> {
    s: Shader,
    x: &'a Parts<'a>,
    bad: &'a HashMap<Value, Why>,
    lane_fail: HashMap<Value, Why>,
    vals: HashMap<Value, Val>,
    floats: HashMap<Value, V>,
    /// DXIL block per IR block (unreachable: none).
    block: Vec<Option<usize>>,
    cur: usize,
    bools: HashMap<(usize, Bool), V>,
    lane_bits: HashMap<Value, V>,
    /// IR phi location by value.
    phi_at: HashMap<Value, (usize, usize)>,
    mask_phis: HashMap<Value, V>,
    pending_masks: Vec<(V, usize, usize)>,
    data_phis: Vec<(V, usize, usize)>,
    read_phis: HashSet<Value>,
    params: HashMap<Value, (usize, u32)>,
    user_data: Option<V>,
    user_reads: BTreeMap<u32, UserData>,
    samplers: Vec<[u32; 4]>,
    descriptors: HashMap<(Site, bool), &'a Source>,
    scalar: HashMap<Site, Address>,
    exports: HashMap<Site, &'a Export>,
    interps: HashMap<Site, &'a Interp>,
    handles: HashMap<(usize, Site, u8, bool, u32), V>,
    /// Blocks splitting the edge from one IR block to another.
    edges: HashMap<(usize, usize), usize>,
    /// Per IR block: a `done` export is behind every lane entering it.
    exported: Vec<bool>,
    /// Reverse-postorder index per IR block.
    rpo: Vec<usize>,
    /// DXIL block each IR block ends in (guarded stores split blocks).
    tail: Vec<Option<usize>>,
    /// The program writes UAVs: barriers fence them too.
    writes_uavs: bool,
    /// src0 of the current instruction as DPP fetched it.
    src0: Option<V>,
    /// Compute code that reads other lanes: thread ids follow the host
    /// lanes, so lane n is console lane n.
    console_lanes: bool,
    /// Mesh: the thread index in the group (made in the entry block).
    flat: Option<V>,
    /// Mesh: IR blocks whose `s_barrier` is left out.
    no_barrier: HashSet<usize>,
    /// Vertex and mesh outputs holding 16-bit values: row, whether two per
    /// dword, and the row packed after it.
    half_outputs: HashMap<u8, (bool, Option<u8>)>,
    /// How lane masks become data.
    mask_data: MaskData,
    /// The instruction being lowered may read lane masks as data.
    mask_ok: bool,
    /// Ballots of masks read as data, by low dword: (low, high).
    ballots: HashMap<Value, (V, V)>,
    /// Masks whose ballot is being built: a mask whose lane bit needs its
    /// own value as a number (an opaque mask keyed by a dword it shares
    /// with another) would otherwise recurse until the stack overflows.
    balloting: HashSet<Value>,
    /// Block writing each value.
    def_block: HashMap<Value, usize>,
    /// IR block being lowered.
    cur_ir: usize,
    /// Whole-wave sections by block and start index.
    sections: HashMap<usize, HashMap<usize, super::wave::Section>>,
    /// Values written in a section that code outside it reads.
    section_out: HashSet<Value>,
    /// The program's constant table (the code from the first byte read
    /// through `s_getpc_b64` on): global, first byte, dwords.
    code_table: Option<(V, u32, u32)>,
    /// Loop headers' iteration counters (see [`LOOP_LIMIT`]): the phi by
    /// IR block, and its incoming (value, DXIL block) pairs.
    counters: HashMap<usize, (V, Vec<(V, usize)>)>,
    /// The block a loop stopped by [`LOOP_LIMIT`] leaves the program from.
    abort: Option<usize>,
    /// A console wave64 program run as two host waves of 32 ([`halves`]).
    halves: Option<halves::Halves>,
    /// See [`Lowered::flagged_accesses`].
    #[cfg(test)]
    flagged_accesses: u32,
    /// See [`Lowered::group_votes`].
    #[cfg(test)]
    group_votes: u32,
}

/// Iterations after which a loop is left (per entry into it), unless
/// [`super::loop_limit`] says otherwise. The console runs a loop as long as
/// its data says; data the host has not mirrored yet (a null view reads
/// zeros) can make that four billion iterations, which hangs the host GPU
/// until the driver resets it. A lane that reaches the limit leaves the
/// program as killed (a pixel discards).
// shortcut: a genuine loop of more iterations than this is cut short too.
pub(super) const LOOP_LIMIT: u32 = 1 << 20;

/// Lane masks read as numbers (`s_bcnt1`, `s_ff1`, `v_mbcnt`, scalar
/// compares, bit tests) are ballots of the lanes' booleans, taken where the
/// mask is written: structurisation's `active` masks hold only lanes that
/// reach that block, so the ballot misses none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaskData {
    /// Compute: host lane n is console lane n (the program declares the
    /// console's wave size), so ballots are the console's masks bit for bit
    /// and any use is exact.
    Console,
    /// Pixel and vertex programs: ballots over the host's wave, whatever its
    /// size. Only uses that mean the same on any wave are allowed: counts,
    /// the first lane, the count below a lane, equality (the HLSL wave
    /// operations the console compiler expands). Lane numbers stay host
    /// lanes throughout (`v_mbcnt`, `v_readlane` at a computed lane).
    Host,
    /// Mesh programs: their lanes and waves (LDS per console wave) are the
    /// console's but no host wave size can be declared.
    None,
}

/// What a value holding a lane mask read as a number stops with.
const LANE_MASK: Why = "a lane mask read as data";

/// Instructions that read a lane mask (not EXEC implicitly, not a lane's
/// bit) as a number.
fn reads_mask_as_data(st: &Structured, i: &Inst) -> bool {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let ops: Vec<&Operand> = match &i.fields {
        // A mask operation (its result a mask) is no data use.
        Fields::Sop { .. }
            if i.writes
                .iter()
                .any(|&(r, w)| matches!(r, Reg::S(_)) && is_mask(w)) =>
        {
            return false;
        }
        Fields::Sop { src, .. } => src.iter().collect(),
        Fields::Valu(v)
            if !matches!(i.name(), "v_cndmask_b32") && !i.name().contains("_co_ci_") =>
        {
            v.src.iter().collect()
        }
        _ => return false,
    };
    ops.into_iter().any(|op| match *op {
        Operand::Scalar { code, .. } if code != NULL => i.read(Reg::S(code)).is_some_and(is_mask),
        _ => false,
    })
}

/// Where half `h` (dword h / 2, the high half when odd) of a parameter
/// packing 16-bit rows `a` and `b` two per dword comes from: a row and
/// channel of `elements`, none for padding. Channel c of `a` is half c;
/// `b` starts at the dword after `a`'s last channel.
fn packed_half(elements: &[SigElement], a: u8, b: Option<u8>, h: u8) -> Option<(u8, u8)> {
    let extent = |r: u8| {
        elements
            .iter()
            .filter(|e| e.row == Some(u32::from(r)))
            .map(|e| e.col + e.cols)
            .max()
    };
    let na = extent(a)?;
    let a_halves = na.div_ceil(2) * 2;
    if h < a_halves {
        return (h < na).then_some((a, h));
    }
    let b = b?;
    let c = h - a_halves;
    (c < extent(b)?).then_some((b, c))
}

fn unsupported(offset: usize, what: Why) -> Stop {
    Stop::Fail(ShaderError::Unsupported { offset, what })
}

/// A broken invariant of the lowering itself: the shader is rejected.
fn internal(what: String) -> Stop {
    Stop::Fail(ShaderError::Internal(what))
}

impl<'a> Lower<'a> {

    /// The DXIL form of an IR value.
    fn get(&mut self, v: Value) -> Result<Val, Why> {
        if self.phi_at.contains_key(&v) {
            self.read_phis.insert(v);
        }
        let is_mask = self.x.st.masks.contains_key(&v) || self.x.st.highs.contains_key(&v);
        match self.vals.get(&v) {
            Some(&Val::Missing(_)) | None if self.mask_ok && is_mask => {
                return self.mask_dword(v).map(Val::I32);
            }
            Some(&Val::Missing(w)) => return Err(w),
            Some(&val) => return Ok(val),
            None => {}
        }
        if let Some(&(param, dword)) = self.params.get(&v) {
            let val = self.user(param, dword);
            self.vals.insert(v, val);
            return Ok(val);
        }
        Err("a value defined after its use")
    }

    fn i32_of(&mut self, v: Value) -> Result<V, Why> {
        match self.get(v)? {
            Val::I32(x) => Ok(x),
            Val::Bit(x) => Ok(self.s.m.cast(self.cur, Cast::ZExt, x, self.s.t.i32)),
            Val::Missing(w) => Err(w),
        }
    }

    /// A user-data dword: a root parameter's value.
    fn user(&mut self, param: usize, dword: u32) -> Val {
        let rp = self.x.native.layout[param];
        match rp.root {
            RootType::ResourceHeap | RootType::SamplerHeap => return Val::I32(self.s.m.i32(0)),
            RootType::Constants | RootType::Table => {}
            _ => return Val::Missing("a root descriptor"),
        }
        let index = u32::from(rp.user_data) + dword;
        self.user_reads.insert(
            index,
            UserData {
                index,
                slot: rp.slot,
                root: rp.root,
                dword,
            },
        );
        self.s.user_data = self.s.user_data.max(index + 1);
        let h = match self.user_data {
            Some(h) => h,
            None => {
                let h = self.s.user_data_handle();
                self.user_data = Some(h);
                h
            }
        };
        let entry = self.s.entry;
        let row = self.s.m.i32(index / 4);
        let r = self.s.op(
            entry,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, row],
        );
        Val::I32(self.s.m.extract(entry, r, index % 4))
    }

    /// A dword of a lane mask as a number: the ballot of the lanes' bits,
    /// taken in the block that writes the mask (see [`MaskData`]).
    fn mask_dword(&mut self, v: Value) -> Result<V, Why> {
        let (lo, high) = match self.x.st.highs.get(&v) {
            Some(&lo) => (lo, true),
            None => (v, false),
        };
        let &m = self.x.st.masks.get(&lo).ok_or(LANE_MASK)?;
        if !self.x.st.active.contains(&lo) {
            return Err("a lane mask holding lanes elsewhere read as data");
        }
        let (x, y) = match self.ballots.get(&lo) {
            Some(&xy) => xy,
            None => {
                let at = match self.def_block.get(&lo) {
                    // Program inputs (EXEC on entry).
                    None => self.s.entry,
                    Some(&b) if b == self.cur_ir => self.cur,
                    Some(&b) => self.tail[b].ok_or("a lane mask written after its use")?,
                };
                if !self.balloting.insert(lo) {
                    return Err("a lane mask whose lanes depend on its own value");
                }
                let bit = self.boolean(m, at);
                self.balloting.remove(&lo);
                let bit = bit?;
                let i32t = self.s.t.i32;
                let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
                let r = self
                    .s
                    .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
                self.s.flags |= flags::WAVE_OPS;
                let mut xy = (self.s.m.extract(at, r, 0), self.s.m.extract(at, r, 1));
                if self.halves.is_some() {
                    // Each half's 32 lanes: the console mask's low and high dwords.
                    xy = self.exchange(at, xy.0);
                }
                self.ballots.insert(lo, xy);
                xy
            }
        };
        Ok(if high { y } else { x })
    }

    // ---- operands ----

    fn operand(&mut self, inst: &Inst, op: &Operand) -> Result<V, Why> {
        match *op {
            Operand::Scalar { code: NULL, .. } => Ok(self.s.m.i32(0)),
            Operand::Scalar { code, .. } => {
                let v = inst.read(Reg::S(code)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            Operand::Vector { reg, .. } => {
                let v = inst.read(Reg::V(reg)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            Operand::Const(c) => crate::resource::inline(c)
                .map(|x| self.s.m.i32(x))
                .ok_or("a special operand (vccz, execz, scc, lds_direct)"),
            Operand::Literal(x) => Ok(self.s.m.i32(x)),
            Operand::Attr { .. } => Err("an attribute operand"),
        }
    }

    /// GFX10 puts an inline f16 constant in the low half of the operand.
    fn f16_operand(&mut self, inst: &Inst, op: &Operand) -> Result<V, Why> {
        match *op {
            Operand::Const(c @ 240..=248) => {
                let h = [
                    0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118,
                ][usize::from(c - 240)];
                Ok(self.s.m.i32(h))
            }
            _ => self.operand(inst, op),
        }
    }

    fn as_float(&mut self, x: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, x, self.s.t.f32)
    }

    fn as_int(&mut self, f: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, f, self.s.t.i32)
    }

    /// The SDWA operand selection of source `k`: a byte or word of the
    /// dword, zero- or sign-extended (none: the whole dword).
    fn sdwa_src(inst: &Inst, k: usize) -> Option<(u8, bool)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        let sel = *s.src_sel.get(k)?;
        let sext = s.sext >> k & 1 != 0;
        (sel < 6).then_some((sel, sext))
    }

    /// Bits `sel` (0-3 a byte, 4-5 a word) of `x`, extended.
    fn select_bits(&mut self, x: V, sel: u8, sext: bool) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8, 8 * sel)
        } else {
            (16, 16 * (sel - 4))
        };
        let left = self.s.m.i32(u32::from(32 - width - at));
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - u32::from(width));
        let op = if sext { Bin::AShr } else { Bin::LShr };
        self.s.m.bin(c, op, t, right)
    }

    /// A float source with its abs/neg modifiers.
    fn fsrc(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let selected = Self::sdwa_src(inst, k);
        let src0 = if k == 0 { self.src0 } else { None };
        let mut f = match (selected, src0, op) {
            (Some((sel, sext)), _, _) => {
                let x = self.operand(inst, &op)?;
                let x = self.select_bits(x, sel, sext);
                self.as_float(x)
            }
            (None, Some(x), _) => self.as_float(x),
            (None, None, Operand::Vector { reg, .. } | Operand::Scalar { code: reg, .. })
                if !matches!(op, Operand::Scalar { code: NULL, .. }) =>
            {
                let r = if matches!(op, Operand::Vector { .. }) {
                    Reg::V(reg)
                } else {
                    Reg::S(reg)
                };
                let val = inst.read(r).ok_or("an unread operand")?;
                match self.floats.get(&val) {
                    Some(&f) => f,
                    None => {
                        let x = self.i32_of(val)?;
                        self.as_float(x)
                    }
                }
            }
            (None, None, _) => {
                let x = self.operand(inst, &op)?;
                self.as_float(x)
            }
        };
        if v.abs >> k & 1 != 0 {
            f = self.unary(6, f);
        }
        if v.neg >> k & 1 != 0 {
            let z = self.s.m.f32(0x8000_0000);
            f = self.s.m.bin(self.cur, Bin::Sub, z, f);
        }
        Ok(f)
    }

    fn src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 {
            return Err("integer operand modifiers");
        }
        match self.src0 {
            Some(x) if k == 0 => Ok(x),
            _ => {
                let x = self.operand(inst, &op)?;
                Ok(match Self::sdwa_src(inst, k) {
                    Some((sel, sext)) => self.select_bits(x, sel, sext),
                    None => x,
                })
            }
        }
    }

    fn unary(&mut self, op: u32, f: V) -> V {
        self.s.op(
            self.cur,
            "unary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[f],
        )
    }

    fn binary(&mut self, op: u32, a: V, b: V) -> V {
        self.s.op(
            self.cur,
            "binary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[a, b],
        )
    }

    /// The hardware's maximum (`max`) or minimum of two floats: as DXIL's
    /// FMax/FMin, but -0 orders below +0 (RDNA2 ISA, `v_max_f32`), where
    /// the compare and select DXIL specifies gives either zero.
    fn max_min(&mut self, max: bool, a: V, b: V) -> V {
        let c = self.cur;
        let r = self.binary(if max { 35 } else { 36 }, a, b);
        let (ia, ib) = (self.as_int(a), self.as_int(b));
        let either = self.s.m.bin(c, Bin::Or, ia, ib);
        let magnitude = self.s.m.i32(0x7FFF_FFFF);
        let either = self.s.m.bin(c, Bin::And, either, magnitude);
        let zero = self.s.m.i32(0);
        let zeros = self.s.m.cmp(c, pred::EQ, either, zero);
        // Of two zeros, +0 unless both are -0 (max), -0 unless neither (min).
        let op = if max { Bin::And } else { Bin::Or };
        let sign = self.s.m.bin(c, op, ia, ib);
        let signed = self.as_float(sign);
        self.s.m.select(c, zeros, signed, r)
    }

    fn fmad(&mut self, a: V, b: V, c: V) -> V {
        self.s.op(
            self.cur,
            "tertiary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            46,
            &[a, b, c],
        )
    }

    /// `(a >> off[4:0]) & ((1 << width) - 1)`, sign-extended from the field's
    /// top bit when `signed`; widths from 32 on take every bit. A signed
    /// field that runs past bit 31 extends from bit 31, as the hardware
    /// does: the shift is arithmetic then.
    fn bfe(&mut self, a: V, off: V, width: V, signed: bool) -> V {
        let c = self.cur;
        let (k31, k32, zero, one, ones) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(0),
            self.s.m.i32(1),
            self.s.m.i32(u32::MAX),
        );
        let off = self.s.m.bin(c, Bin::And, off, k31);
        let shift = if signed { Bin::AShr } else { Bin::LShr };
        let shifted = self.s.m.bin(c, shift, a, off);
        let full = self.s.m.cmp(c, pred::IUGE, width, k32);
        let w = self.s.m.bin(c, Bin::And, width, k31);
        let mask = self.s.m.bin(c, Bin::Shl, one, w);
        let mask = self.s.m.bin(c, Bin::Sub, mask, one);
        let mask = self.s.m.select(c, full, ones, mask);
        let r = self.s.m.bin(c, Bin::And, shifted, mask);
        if !signed {
            return r;
        }
        // Sign-extend a field narrower than 32 bits (none for width 0).
        let left = self.s.m.bin(c, Bin::Sub, k32, w);
        let left = self.s.m.bin(c, Bin::And, left, k31);
        let t = self.s.m.bin(c, Bin::Shl, r, left);
        let t = self.s.m.bin(c, Bin::AShr, t, left);
        let is0 = self.s.m.cmp(c, pred::EQ, w, zero);
        let keep = self.s.m.bin(c, Bin::Or, full, is0);
        self.s.m.select(c, keep, r, t)
    }

    /// The median of three. With a NaN among them the hardware gives the
    /// minimum of the three instead (the host's minimum skips NaNs, so that
    /// is the smallest number among them).
    fn med3(&mut self, a: V, b: V, third: V) -> V {
        let c = self.cur;
        let lo = self.binary(36, a, b);
        let hi = self.binary(35, a, b);
        let t = self.binary(36, hi, third);
        let median = self.binary(35, lo, t);
        let least = self.binary(36, lo, third);
        let ab = self.s.m.cmp(c, pred::UNO, a, b);
        let nan3 = self.s.m.cmp(c, pred::UNO, third, third);
        let nan = self.s.m.bin(c, Bin::Or, ab, nan3);
        self.s.m.select(c, nan, least, median)
    }

    /// `a - floor(a)`, which on the hardware never reaches 1.0: where it
    /// would round to it (a tiny negative `a`), the result is `below_one`,
    /// the largest value under 1.0 in the operation's width. Hosts differ
    /// here (some give 1.0). Comparing against a `threshold` distinct from
    /// `below_one` prevents drivers (observed on NVIDIA) from folding
    /// `r > k ? k : r` into `minNum` (`FMNMX`) and swallowing NaN.
    fn fract(&mut self, a: V, threshold: f32, below_one: f32) -> V {
        let c = self.cur;
        let r = self.unary(22, a);
        let limit = self.s.m.f32(threshold.to_bits());
        let k = self.s.m.f32(below_one.to_bits());
        let over = self.s.m.cmp(c, pred::OGE, r, limit);
        self.s.m.select(c, over, k, r)
    }

    /// Sine, or cosine, of an angle in turns, as the hardware takes it. The
    /// whole turns are taken off first (`x - round_even(x)`, exact, within
    /// half a turn), so a large angle keeps its fraction; scaling to radians
    /// first rounds it away. A zero passes unreduced (the sine of -0 is -0).
    /// The sine of a half turn and the cosine of a quarter turn are exactly
    /// +0 on the hardware, where the host's function of the rounded angle
    /// gives about 1e-7. Infinities and NaN give NaN.
    /// shortcut: below 2^-12 turns the hardware scales by a slightly
    /// different constant than 2π (a last-place difference).
    fn trig_turns(&mut self, x: V, cosine: bool) -> V {
        let c = self.cur;
        let zero = self.s.m.f32(0);
        let whole = self.unary(26, x);
        let part = self.s.m.bin(c, Bin::Sub, x, whole);
        let is_zero = self.s.m.cmp(c, pred::OEQ, x, zero);
        let part = self.s.m.select(c, is_zero, x, part);
        let k = self.s.m.f32(TWO_PI.to_bits());
        let angle = self.s.m.bin(c, Bin::Mul, part, k);
        let r = self.unary(if cosine { 12 } else { 13 }, angle);
        let at = if cosine { 0.25f32 } else { 0.5 };
        let at = self.s.m.f32(at.to_bits());
        let size = self.unary(6, part);
        let exact = self.s.m.cmp(c, pred::OEQ, size, at);
        self.s.m.select(c, exact, zero, r)
    }

    /// The high dword of a 32 × 32-bit product.
    fn mul_hi(&mut self, a: V, b: V, signed: bool) -> V {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let ext = if signed { Cast::SExt } else { Cast::ZExt };
        let a = self.s.m.cast(c, ext, a, i64t);
        let b = self.s.m.cast(c, ext, b, i64t);
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let k32 = self.s.m.int_const(i64t, 32);
        let hi = self.s.m.bin(c, Bin::LShr, m, k32);
        self.s.flags |= flags::INT64;
        self.s.m.cast(c, Cast::Trunc, hi, self.s.t.i32)
    }

    // ---- lane booleans ----

    fn boolean(&mut self, b: Bool, at: usize) -> Result<V, Why> {
        // Only the two constants have no decision node.
        let Some((atom, lo, hi)) = self.x.st.bools.node(b) else {
            return Ok(self.s.m.bool(b == Bool::TRUE));
        };
        if let Some(&v) = self.bools.get(&(at, b)) {
            return Ok(v);
        }
        let a = self.atom(atom, at)?;
        let m = &mut self.s.m;
        let v = match (lo, hi) {
            (Bool::FALSE, Bool::TRUE) => a,
            (Bool::TRUE, Bool::FALSE) => {
                let t = m.bool(true);
                m.bin(at, Bin::Xor, a, t)
            }
            (Bool::FALSE, h) => {
                let h = self.boolean(h, at)?;
                self.s.m.bin(at, Bin::And, a, h)
            }
            (l, Bool::TRUE) => {
                let l = self.boolean(l, at)?;
                self.s.m.bin(at, Bin::Or, a, l)
            }
            (l, h) => {
                let l = self.boolean(l, at)?;
                let h = self.boolean(h, at)?;
                self.s.m.select(at, a, h, l)
            }
        };
        self.bools.insert((at, b), v);
        Ok(v)
    }

    fn atom(&mut self, atom: Atom, at: usize) -> Result<V, Why> {
        match atom {
            Atom::Lane(v) => self.lane_bits.get(&v).copied().ok_or_else(|| {
                self.lane_fail
                    .get(&v)
                    .copied()
                    .unwrap_or("a lane mask from a carry")
            }),
            Atom::Uniform(v) => match self.get(v)? {
                Val::Bit(x) => Ok(x),
                Val::I32(x) => {
                    let z = self.s.m.i32(0);
                    Ok(self.s.m.cmp(at, pred::NE, x, z))
                }
                Val::Missing(w) => Err(w),
            },
            Atom::Phi(id) => {
                if let Some(&v) = self.mask_phis.get(&id) {
                    return Ok(v);
                }
                let &(b, i) = self
                    .phi_at
                    .get(&id)
                    .ok_or("a lane-mask join without a phi")?;
                let blk = self.block[b].ok_or("an unreachable join")?;
                let v = self.s.m.phi(blk, self.s.t.i1);
                self.mask_phis.insert(id, v);
                self.pending_masks.push((v, b, i));
                Ok(v)
            }
            // shortcut: "some lane of the quad is set" is taken as true: the
            // lanes it adds only feed derivatives, and exports keep their own
            // masks. Wrong if such a section had a side effect.
            Atom::Quad(_) => Ok(self.s.m.bool(true)),
            // With console lanes, a mask held as a number is the console's:
            // the lane tests its own bit.
            Atom::Opaque(v) if self.console_lanes => {
                let hi = match self.pair_high(v) {
                    Some(h) => Some(self.get(h)?),
                    None => None,
                };
                let lo = self.get(v)?;
                let word = |w: Val| match w {
                    Val::I32(x) => Ok(x),
                    _ => Err("an opaque lane mask"),
                };
                let lo = word(lo)?;
                let hi = hi.map(word).transpose()?;
                Ok(self.lane_bit(at, lo, hi))
            }
            Atom::Constant(k) if self.console_lanes => {
                let lo = self.s.m.i32(k as u32);
                let hi = self.s.m.i32((k >> 32) as u32);
                Ok(self.lane_bit(at, lo, Some(hi)))
            }
            Atom::Opaque(_) => Err("an opaque lane mask"),
            Atom::Constant(_) => Err("a constant lane mask"),
        }
    }

    /// This lane's bit of the mask `lo`, `hi` (wave64).
    fn lane_bit(&mut self, at: usize, lo: V, hi: Option<V>) -> V {
        let lane = if self.halves.is_some() {
            self.console_lane()
        } else {
            self.s.op(
                at,
                "waveGetLaneIndex",
                Attr::ReadOnly,
                self.s.t.i32,
                111,
                &[],
            )
        };
        let (k31, k32, one, z) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let word = match hi {
            Some(hi) => {
                let low = self.s.m.cmp(at, pred::IULT, lane, k32);
                self.s.m.select(at, low, lo, hi)
            }
            None => lo,
        };
        let sh = self.s.m.bin(at, Bin::And, lane, k31);
        let t = self.s.m.bin(at, Bin::LShr, word, sh);
        let t = self.s.m.bin(at, Bin::And, t, one);
        self.s.m.cmp(at, pred::NE, t, z)
    }

    /// The value of the high dword of the wave64 register pair whose low
    /// dword holds `lo` (none in wave32).
    fn pair_high(&self, lo: Value) -> Option<Value> {
        if !self.x.native.wave64 {
            return None;
        }
        let p = self.x.p;
        let pick = |reg: Reg| match reg {
            Reg::S(c) => Some(Reg::S(c + 1)),
            _ => None,
        };
        if let Some(&(reg, _)) = p.inputs.iter().find(|i| i.1 == lo) {
            let h = pick(reg)?;
            return p.inputs.iter().find(|i| i.0 == h).map(|i| i.1);
        }
        if let Some(&(b, i)) = self.phi_at.get(&lo) {
            let h = pick(p.blocks[b].phis[i].reg)?;
            return p.blocks[b]
                .phis
                .iter()
                .find(|q| q.reg == h)
                .map(|q| q.value);
        }
        let b = *self.def_block.get(&lo)?;
        let inst = p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == lo))?;
        let &(reg, _) = inst.writes.iter().find(|w| w.1 == lo)?;
        inst.write(pick(reg)?)
    }

    fn guard(&mut self, b: usize, i: usize) -> Result<Option<V>, Why> {
        let g = match self.x.st.guards[b][i] {
            None => None,
            Some(g) => Some(self.boolean(g, self.cur)?),
        };
        Ok(self.group_flag(b, g))
    }

    // ---- results ----

    fn set(&mut self, v: Option<Value>, val: Val) {
        if let Some(v) = v {
            self.vals.insert(v, val);
        }
    }

    /// Writes VGPR `reg` (or the SGPR of a lane read), under `guard`.
    fn write(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let x = match (Self::sdwa_dst(inst), reg) {
            (Some((sel, unused)), Reg::V(_)) => self.place_bits(inst, reg, x, sel, unused),
            _ => x,
        };
        self.write_placed(inst, reg, x, guard);
    }

    /// [`Self::write`] of a dword already placed (SDWA applied).
    fn write_placed(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let Some(new) = inst.write(reg) else { return };
        let x = match (guard, reg) {
            (Some(g), Reg::V(_)) => {
                let old = match inst.read(reg).map(|o| self.i32_of(o)) {
                    Some(Ok(o)) => o,
                    // The validator refuses reads of undef.
                    _ => self.s.m.i32(0),
                };
                self.s.m.select(self.cur, g, x, old)
            }
            _ => x,
        };
        self.vals.insert(new, Val::I32(x));
    }

    /// The SDWA destination selection: a byte or word, and what the other
    /// bits get (0 zeros, 1 the sign above the field, 2 the old bits).
    fn sdwa_dst(inst: &Inst) -> Option<(u8, u8)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        (s.dst_sel < 6).then_some((s.dst_sel, s.dst_unused))
    }

    /// The low byte or word of `x` placed at `sel` in VGPR `reg`.
    fn place_bits(&mut self, inst: &Inst, reg: Reg, x: V, sel: u8, unused: u8) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8u32, 8 * u32::from(sel))
        } else {
            (16, 16 * u32::from(sel - 4))
        };
        let left = self.s.m.i32(32 - width);
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - width - at);
        if unused == 1 {
            // Bits below the field shift in as zeros.
            return self.s.m.bin(c, Bin::AShr, t, right);
        }
        let field = self.s.m.bin(c, Bin::LShr, t, right);
        if unused != 2 {
            return field;
        }
        let old = match inst.read(reg).map(|o| self.i32_of(o)) {
            Some(Ok(o)) => o,
            _ => self.s.m.i32(0),
        };
        let keep = self.s.m.i32(!((((1u64 << width) - 1) as u32) << at));
        let old = self.s.m.bin(c, Bin::And, old, keep);
        self.s.m.bin(c, Bin::Or, old, field)
    }

    fn write_f(&mut self, inst: &Inst, f: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let mut f = f;
        match v.omod {
            0 => {}
            k => {
                let factor = [1.0f32, 2.0, 4.0, 0.5][usize::from(k)];
                let c = self.s.m.f32(factor.to_bits());
                f = self.s.m.bin(self.cur, Bin::Mul, f, c);
            }
        }
        if v.clamp {
            f = self.unary(7, f);
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a float result outside a VGPR");
        };
        let x = self.as_int(f);
        self.write(inst, Reg::V(reg), x, guard);
        if guard.is_none()
            && Self::sdwa_dst(inst).is_none()
            && let Some(new) = inst.write(Reg::V(reg))
        {
            self.floats.insert(new, f);
        }
        Ok(())
    }

    fn write_i(&mut self, inst: &Inst, x: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.clamp || v.omod != 0 {
            return Err("integer output modifiers");
        }
        match v.dst {
            Some(Operand::Vector { reg, .. }) => self.write(inst, Reg::V(reg), x, guard),
            Some(Operand::Scalar { code, .. }) => self.write(inst, Reg::S(code), x, None),
            _ => return Err("a result without a register"),
        }
        Ok(())
    }

    /// Starts code only the lanes where `guard` holds run (a store under a
    /// lane guard): returns where it started and the join block.
    fn begin_guarded(&mut self, guard: Option<V>) -> Option<(usize, usize)> {
        let g = guard?;
        let (then, join) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(self.cur, Term::CondBr(g, then, join));
        let head = self.cur;
        self.cur = then;
        Some((head, join))
    }

    /// Ends guarded code; each (value inside, value outside) becomes a phi.
    fn end_guarded(&mut self, at: Option<(usize, usize)>, values: &[(V, V)]) -> Vec<V> {
        let Some((head, join)) = at else {
            return values.iter().map(|v| v.0).collect();
        };
        let then = self.cur;
        self.s.m.terminate(then, Term::Br(join));
        self.cur = join;
        values
            .iter()
            .map(|&(inside, outside)| {
                let ty = self.s.m.type_of(inside);
                let phi = self.s.m.phi(join, ty);
                self.s.m.set_phi(phi, vec![(inside, then), (outside, head)]);
                phi
            })
            .collect()
    }

    /// Code that depends on the lane layout: wave operations, and in compute
    /// the console's wave size (lanes are thread ids in group order there).
    /// Pixel lanes match only within a quad.
    fn cross_lane(&mut self, quad: bool) -> Result<(), Why> {
        if self.s.mesh.is_some() {
            // shortcut: no wave size for mesh shaders in SM 6.6, so host
            // lanes cannot follow the console's; emulate through group-shared
            // memory if mesh programs turn out to need these.
            return Err("a cross-lane operation in a mesh shader");
        }
        if self.s.threads.is_none() && !(quad && self.s.pixel) {
            return Err("a cross-lane operation outside compute");
        }
        debug_assert!(self.s.threads.is_none() || self.console_lanes);
        self.s.flags |= flags::WAVE_OPS;
        Ok(())
    }

    /// The console wave size.
    fn wave(&self) -> u32 {
        if self.x.native.wave64 { 64 } else { 32 }
    }

    fn lane_index(&mut self) -> V {
        if self.s.mesh.is_some() {
            // The console's lane: mesh code reads no other lanes, so host
            // waves need not match the console's.
            let f = self.mesh_thread();
            let m = self.s.m.i32(self.wave() - 1);
            return self.s.m.bin(self.s.entry, Bin::And, f, m);
        }
        if self.halves.is_some() {
            return self.console_lane();
        }
        self.s.op(
            self.cur,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.s.t.i32,
            111,
            &[],
        )
    }

    /// `x` of lane `src` (in this lane's quad) through quad reads, whose
    /// lane must be a constant.
    fn quad_pick(&mut self, x: V, src: V) -> V {
        self.s.flags |= flags::WAVE_OPS;
        let c = self.cur;
        let three = self.s.m.i32(3);
        let q = self.s.m.bin(c, Bin::And, src, three);
        let mut r = x;
        for n in 0..4 {
            let kn = self.s.m.i32(n);
            let read = self.s.op(
                c,
                "quadReadLaneAt.i32",
                Attr::NoUnwind,
                self.s.t.i32,
                122,
                &[x, kn],
            );
            let is = self.s.m.cmp(c, pred::EQ, q, kn);
            r = self.s.m.select(c, is, read, r);
        }
        r
    }

    fn read_lane(&mut self, x: V, lane: V) -> V {
        // In halves, console lane `lane` is host lane `lane & 31` of the half
        // holding it; DPP and `ds_swizzle` stay in the reading lane's half.
        let lane = if self.halves.is_some() {
            let k31 = self.s.m.i32(31);
            self.s.m.bin(self.cur, Bin::And, lane, k31)
        } else {
            lane
        };
        self.s.op(
            self.cur,
            "waveReadLaneAt.i32",
            Attr::NoUnwind,
            self.s.t.i32,
            117,
            &[x, lane],
        )
    }

    /// The float value of a register value.
    fn float_of(&mut self, v: Value) -> Result<V, Why> {
        if let Some(&f) = self.floats.get(&v) {
            return Ok(f);
        }
        let x = self.i32_of(v)?;
        Ok(self.as_float(x))
    }

    /// The values of VGPRs `reg..reg + n` as `inst` reads them.
    fn vgprs(&mut self, inst: &Inst, op: &Operand, n: u8) -> Result<Vec<V>, Why> {
        let Operand::Vector { reg, .. } = *op else {
            return Err("a vector operand");
        };
        (0..n)
            .map(|k| {
                let v = inst.read(Reg::V(reg + k)).ok_or("an unread operand")?;
                self.i32_of(v)
            })
            .collect()
    }

    // ---- instructions ----

    fn inst(&mut self, b: usize, i: usize) -> Result<(), Stop> {
        let inst: &'a Inst = &self.x.p.blocks[b].insts[i];
        let offset = inst.machine.offset;
        let name = inst.name();
        self.mask_ok = match self.mask_data {
            MaskData::Console => true,
            MaskData::Host => {
                name.starts_with("s_bcnt1_i32_")
                    || name.starts_with("s_ff1_i32_")
                    || name.starts_with("v_mbcnt_")
                    || matches!(
                        name,
                        "s_cmp_eq_u32" | "s_cmp_lg_u32" | "s_cmp_eq_u64" | "s_cmp_lg_u64"
                    )
            }
            MaskData::None => false,
        };
        let effect = matches!(inst.fields, Fields::Exp { .. })
            || matches!(inst.fields, Fields::Flat { .. }) && !name.starts_with("global_load_")
            || name.contains("store")
            || name.contains("atomic")
            || name == "s_barrier"
            || matches!(inst.fields, Fields::Ds { .. })
                && !(name.starts_with("ds_read")
                    || name.starts_with("ds_swizzle")
                    || name.contains("permute"));
        let r = match &inst.fields {
            Fields::Sop { .. } => self.sop(inst),
            Fields::Sopp { .. } if name == "s_barrier" && self.no_barrier.contains(&b) => Ok(()),
            Fields::Sopp { .. } if name == "s_barrier" => {
                // Thread-group sync with a shared-memory fence, and a UAV
                // fence when the program writes UAVs (the waits before an
                // s_barrier make those writes visible on the console).
                let mode = if self.writes_uavs { 1 | 2 | 8 } else { 1 | 8 };
                let mode = self.s.m.i32(mode);
                let void = self.s.t.void;
                self.s
                    .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
                Ok(())
            }
            Fields::Sopp { .. } => Ok(()),
            Fields::Smem { .. } => self.flagged(b, inst, |l| l.smem(inst, (b, i))),
            Fields::Valu(_) if name.starts_with("v_interp_") => self.interp(inst, (b, i)),
            Fields::Valu(_) => self.valu(inst, b, i),
            Fields::Vintrp { .. } => self.interp(inst, (b, i)),
            Fields::Image { .. } => self.flagged(b, inst, |l| l.image(inst, b, i)),
            Fields::Buffer { .. } => self.flagged(b, inst, |l| l.buffer(inst, b, i)),
            Fields::Exp { .. } => self.export(b, i),
            Fields::Ds { .. } => self.ds(inst, b, i),
            Fields::Flat { .. } if name.starts_with("global_") => {
                self.flagged(b, inst, |l| l.flat(inst, b, i))
            }
            Fields::Flat { .. } => Err("flat memory"),
        };
        match r {
            Ok(()) => Ok(()),
            Err(why) if effect => Err(unsupported(offset, why)),
            Err(why) => {
                for &(_, v) in &inst.writes {
                    self.vals.insert(v, Val::Missing(why));
                }
                Ok(())
            }
        }
    }

    /// `x` (an f16 in the low bits) as the low half of VGPR `reg`: 16-bit
    /// results keep the register's high half (GFX10).
    fn low_half(&mut self, inst: &Inst, reg: u8, x: V) -> V {
        let c = self.cur;
        let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
            Some(Ok(o)) => o,
            _ => self.s.m.i32(0),
        };
        let (high, low) = (self.s.m.i32(0xFFFF_0000), self.s.m.i32(0xFFFF));
        let old = self.s.m.bin(c, Bin::And, old, high);
        let x = self.s.m.bin(c, Bin::And, x, low);
        self.s.m.bin(c, Bin::Or, old, x)
    }
}

#[cfg(test)]
mod tests {
    use super::{Lowered, Parts, lower};
    use crate::native::{
        ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
    };
    use crate::{io, ir, resource, structure};

    const S_ENDPGM: u32 = 0xBF81_0000;

    /// Whether a synthetic compute program translates with these input registers.
    fn lowers(words: &[u32], rsrc2: u32) -> Result<(), crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2,
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, false, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            host_lanes: crate::emit::CONSOLE_WAVE,
        })
        .map(|_| ())
    }

    /// Lowers a compute program with the root parameters `layout` (in user
    /// SGPRs from s0 on) and the used resources `used`.
    fn lower_with(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
    ) -> Result<Lowered, crate::ShaderError> {
        lower_wave(words, layout, used, false)
    }

    /// [`lower_with`] in wave64 if `wave64`.
    fn lower_wave(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
        wave64: bool,
    ) -> Result<Lowered, crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64,
            vgprs: 256,
            layout,
            used,
            io: StageIo::Cs(ComputeIo {
                // The layout's user SGPRs (at least two), 512 bytes of LDS.
                rsrc2: 1 << 15 | sgprs.max(2) << 1,
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, wave64, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
        let io = io::raise(&p, &native).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            host_lanes: crate::emit::CONSOLE_WAVE,
        })
    }

    /// Root parameter `slot` in user SGPRs from `user_data` on.
    fn root(root: RootType, user_data: u8, slot: u8) -> RootParam {
        RootParam {
            root,
            table_offset_dwords: 0,
            sgprs: if root == RootType::Descriptor { 2 } else { 1 },
            base_register: 0,
            user_data,
            slot,
        }
    }

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

    fn perm_features(selector: u32) -> u64 {
        // Synthetic compute shader:
        // v0 is the thread ID and can supply a dynamic selector.
        // Store the result in LDS to keep the permutation live.
        let words = [
            0x7E02_02FFu32,
            0x817F_34AB,
            0x7E04_02FF,
            0xFE01_56CD,
            0xD744_0003,
            257 | (258 << 9) | (selector << 18),
            0x3408_0082,
            0xD834_0000,
            0x0000_0304,
            S_ENDPGM,
        ];

        let code: Vec<u8> = words.iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();

        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 15,
                threads: [64, 1, 1],
            }),
            code,
        };

        let p = ir::build(&native.code, false, 256).unwrap();

        assert!(p.blocks.iter()
            .flat_map(|b| &b.insts)
            .any(|i| i.name() == "v_perm_b32"));

        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();

        let io = io::raise(&p, &native).unwrap();

        let emitted = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            host_lanes: crate::emit::CONSOLE_WAVE,
        }).unwrap();

        let container =
            crate::Container::parse(&emitted.dxil).unwrap();

        let features =
            container.part(crate::FourCC::SFI0).unwrap();

        u64::from_le_bytes(features.data.try_into().unwrap())
    }

    #[test]
    fn perm_constant_selector_needs_no_int64() {
        assert_eq!(perm_features(128) & 0x8000, 0);
    }

    #[test]
    fn perm_dynamic_selector_needs_no_int64() {
        assert_eq!(
            perm_features(256) & 0x8000,
            0,
            "dynamic v_perm_b32 must not require INT64"
        );
    }

    #[test]
    fn a_descriptor_selected_by_scc_picks_its_heap_index() {
        // Two buffer descriptors loaded from the table at s0, one picked by
        // a root constant (s1) with s_cselect_b64 per half, and loaded
        // through by index and stored to LDS so the load is needed.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xF408_0302,
            0xFA00_0010, // s_load_dwordx4 s[12:15], s[4:5], 0x10
            0xBF06_8001, // s_cmp_eq_u32 s1, 0
            0x8590_0C08, // s_cselect_b64 s[16:17], s[8:9], s[12:13]
            0x8592_0E0A, // s_cselect_b64 s[18:19], s[10:11], s[14:15]
            0xE030_2000,
            0x8004_0100, // buffer_load_dword v1, v0, s[16:19], 0 idxen
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let layout = || vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
        // Both are structured buffers of dwords in the table's range.
        let used = || {
            vec![UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 4,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }]
        };
        lower_with(&words, layout(), used()).unwrap();
    }

    /// Tables of 2D textures at s0 and samplers at s1.
    fn texture_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
        let range = |kind, user_data, slot| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: if kind == ResourceType::Sampler { 4 } else { 8 },
            user_data,
            slot,
        };
        (
            vec![root(RootType::Table, 0, 0), root(RootType::Table, 1, 1)],
            vec![
                range(ResourceType::Srv, 0, 0),
                range(ResourceType::Sampler, 1, 1),
            ],
        )
    }

    #[test]
    fn waterfall_loops_over_per_lane_descriptors_sample_with_each_lanes_own() {
        use crate::resource::tests::{
            WATERFALL_FETCHED_WAVE32, WATERFALL_FETCHED_WAVE64, WATERFALL_PICKED_WAVE32,
        };
        for (words, wave64) in [
            (WATERFALL_FETCHED_WAVE32, false),
            (WATERFALL_FETCHED_WAVE64, true),
            (WATERFALL_PICKED_WAVE32, false),
        ] {
            let (layout, used) = texture_tables();
            let l = lower_wave(words, layout, used, wave64).unwrap();
            // The loop's reads of the first lane became each lane's own
            // dwords: no wave operation is left.
            assert_eq!(l.nonuniform_handles, 1);
            assert!(!l.wave_ops);
        }
    }

    /// [`texture_tables`] with a raw buffer UAV at dword 16 of the first.
    pub(crate) fn texture_and_uav_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
        let (layout, mut used) = texture_tables();
        used.push(UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 16,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        });
        (layout, used)
    }

    #[test]
    fn waterfall_loops_over_descriptors_joined_per_lane_sample_with_each_lanes_own() {
        use crate::resource::tests::{WATERFALL_JOINED_MIXED_WAVE32, WATERFALL_JOINED_WAVE32};
        for words in [WATERFALL_JOINED_WAVE32, WATERFALL_JOINED_MIXED_WAVE32] {
            let (layout, used) = texture_and_uav_tables();
            let l = lower_wave(words, layout, used, false).unwrap();
            // One image handle, from the join's per-lane heap index.
            assert_eq!(l.nonuniform_handles, 1);
            assert!(!l.wave_ops);
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

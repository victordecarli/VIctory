//! Console wave64 compute programs on a host whose waves have 32 lanes: which ones run as
//! two host waves ("halves", `lower::halves`), and, on a host GPU that takes waves of 32,
//! that a split program computes what one console wave of 64 does. Expected values come
//! from the GFX10 instructions' definitions for a 64-lane wave, chosen so that each half
//! computing on its own gives a different answer.
//!
//! The program shape (root constants in s0..s3, a raw UAV's descriptor loaded into
//! s[8:11]) follows the wave64 tests of the fxceless line (`base/fxceless`).

use super::{Container, HostShader, ShaderError};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

// The body's building blocks; `encodings_decode_as_named` checks each.
/// v_mbcnt_lo_u32_b32 v1, -1, 0; v_mbcnt_hi_u32_b32 v1, -1, v1: the console lane.
const MBCNT: [u32; 4] = [0xD765_0001, 0x0001_00C1, 0xD766_0001, 0x0002_02C1];
/// v_cmp_le_u32 vcc, 40, v0: the lanes from 40 on.
const VCC_FROM_40: u32 = 0x7D86_00A8;
/// v_mov_b32 v1, vcc_hi
const V1_VCC_HI: u32 = 0x7E02_026B;
/// s_bcnt1_i32_b64 s12, vcc
const BCNT_VCC: u32 = 0xBE8C_106A;
/// v_mov_b32 v1, s12
const V1_S12: u32 = 0x7E02_020C;
/// v_readlane_b32 s12, v0, 40
const READLANE_40: [u32; 2] = [0xD760_000C, 0x0001_5100];
/// v_readfirstlane_b32 s12, v0
const READFIRSTLANE: u32 = 0x7E18_0500;
/// v_readfirstlane_b32 s14, v0
const READFIRSTLANE_V0_S14: u32 = 0x7E1C_0500;
/// v_readlane_b32 s14, v0, 10
const READLANE_10: [u32; 2] = [0xD760_000E, 0x0001_1500];
/// v_mov_b32 v1, v0
const V1_V0: u32 = 0x7E02_0300;
/// v1 = v0; each lane adds lane 10's v0 (read into s14 in each iteration) to v1 until it
/// reaches 40 (EXEC saved in s[12:13]).
const LANE_LOOP: [u32; 9] = [
    V1_V0,
    0xBE8C_047E, // s_mov_b64 s[12:13], exec
    READLANE_10[0],
    READLANE_10[1],
    vop2(0x25, 1, 14, 1), // v_add_nc_u32 v1, s14, v1
    0x7D86_02A8,          // v_cmp_le_u32 vcc, 40, v1
    0x8AFE_6A7E,          // s_andn2_b64 exec, exec, vcc
    0xBF89_FFFA,          // s_cbranch_execnz back to the v_readlane
    END_IF,
];
/// v1 = v0; each round, v1 += lane 10's v0 (read into s14), and the lanes that reached 40
/// leave (`break`: the loop's first exit, when every lane left); then v1 += 1, and the
/// lanes that reached 40 leave at the loop's second exit.
const LOOP_TWO_EXITS: [u32; 13] = [
    V1_V0,
    0xBE8C_047E, // s_mov_b64 s[12:13], exec
    READLANE_10[0],
    READLANE_10[1],
    vop2(0x25, 1, 14, 1),   // v_add_nc_u32 v1, s14, v1
    0x7D86_02A8,            // v_cmp_le_u32 vcc, 40, v1
    0x8AFE_6A7E,            // s_andn2_b64 exec, exec, vcc
    0xBF88_0004,            // s_cbranch_execz past the loop
    vop2(0x25, 1, 0x81, 1), // v_add_nc_u32 v1, 1, v1
    0x7D86_02A8,            // v_cmp_le_u32 vcc, 40, v1
    0x8AFE_6A7E,            // s_andn2_b64 exec, exec, vcc
    0xBF89_FFF6,            // s_cbranch_execnz back to the v_readlane
    END_IF,
];
/// v1 = v0; for the lanes from 20 on, three rounds of a loop on a scalar counter (s15) in
/// which the lanes from 40 on add lane 10's v0 (read into s14) to v1. The two ways through
/// the inner branch restore EXEC differently, so the loop's latch joins EXEC with a phi,
/// which the loop's header joins again over its back edge.
const UNIFORM_LOOP_IN_A_BRANCH: [u32; 18] = [
    V1_V0,
    0x7D86_0094, // v_cmp_le_u32 vcc, 20, v0
    0xBE92_246A, // s_and_saveexec_b64 s[18:19], vcc
    0xBF88_000D, // s_cbranch_execz past the loop
    0xBE8F_0380, // s_mov_b32 s15, 0
    VCC_FROM_40, // the loop's header
    0xBE90_246A, // s_and_saveexec_b64 s[16:17], vcc
    0xBF88_0005, // s_cbranch_execz to the second restore
    READLANE_10[0],
    READLANE_10[1],
    vop2(0x25, 1, 14, 1), // v_add_nc_u32 v1, s14, v1
    0x88FE_107E,          // s_or_b64 exec, exec, s[16:17]
    0xBF82_0001,          // s_branch to the latch
    RESTORE,              // s_mov_b64 exec, s[16:17]
    0x810F_810F,          // the latch: s_add_i32 s15, s15, 1
    0xBF0A_830F,          // s_cmp_lt_u32 s15, 3
    0xBF85_FFF4,          // s_cbranch_scc1 to the header
    0xBEFE_0412,          // s_mov_b64 exec, s[18:19]
];
/// v_mov_b32 v1, 0; s_mov_b32 s12, 7; v_writelane_b32 v1, s12, 40
const WRITELANE_40: [u32; 4] = [0x7E02_0280, 0xBE8C_0387, 0xD761_0001, 0x0001_500C];
/// v_lshlrev_b32 v2, 2, v0; buffer_store_dword v1, v2, s[8:11], 0 offen: v1 to dword
/// (thread index) of the UAV.
const STORE_V1: [u32; 3] = [0x3404_0082, 0xE070_1000, 0x8002_0102];
/// s_and_saveexec_b64 s[12:13], vcc; s_cbranch_execz over one dword
const IF_VCC: [u32; 2] = [0xBE8C_246A, 0xBF88_0001];
/// s_mov_b64 exec, s[12:13]
const END_IF: u32 = 0xBEFE_040C;
/// s_barrier
const BARRIER: u32 = 0xBF8A_0000;
/// s_mov_b32 s14, 0 / s_mov_b32 s14, 5
const S14_0: u32 = 0xBE8E_0380;
const S14_5: u32 = 0xBE8E_0385;
/// s_mov_b32 s14, 7; v_mov_b32 v1, s14
const S14_7: u32 = 0xBE8E_0387;
const V1_S14: u32 = 0x7E02_020E;
/// Only the lanes from 40 on: s14 = 7, v1 = s14 (the branch skips the two dwords).
const IF_FROM_40_V1_7: [u32; 6] = [VCC_FROM_40, IF_VCC[0], 0xBF88_0002, S14_7, V1_S14, END_IF];
/// ds_write_b32 v2, v1
const DS_WRITE: [u32; 2] = [0xD834_0000, 0x0000_0102];
/// v_lshlrev_b32 v2, 2, v0; v_mov_b32 v1, v0; ds_write_b32 v2, v1: thread t writes t to
/// LDS dword t.
const LDS_WRITE_T: [u32; 4] = [0x3404_0082, 0x7E02_0300, DS_WRITE[0], DS_WRITE[1]];
/// v_sub_nc_u32 v3, 63, v0; v_lshlrev_b32 v3, 2, v3; ds_read_b32 v1, v3: thread t reads
/// LDS dword 63 - t, which the other half wrote.
const LDS_READ_MIRROR: [u32; 4] = [0x4C06_00BF, 0x3406_0682, 0xD8D8_0000, 0x0100_0003];

const fn vop1(op: u32, vdst: u32, src0: u32) -> u32 {
    0x3F << 25 | vdst << 17 | op << 9 | src0
}

const fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
    op << 25 | vdst << 17 | vsrc1 << 9 | src0
}

/// `v_<op> vdst, vsrc, vdst` with DPP16 `ctrl` on `vsrc`, every row and bank, `bound_ctrl:1`.
const fn dpp(op: u32, vdst: u32, vsrc: u32, ctrl: u32) -> [u32; 2] {
    [vop2(op, vdst, 0xFA, vdst), vsrc | ctrl << 8 | 1 << 19 | 0xFF << 24]
}

/// `v_readlane_b32 s<sdst>, v1, lane`
const fn readlane_v1(sdst: u32, lane: u32) -> [u32; 2] {
    [0xD760_0000 | sdst, 0x101 | (128 + lane) << 9]
}

const V_OR: u32 = 0x1C;
const V_AND: u32 = 0x1B;
const V_ADD_U: u32 = 0x25;
const V_MIN_U: u32 = 0x13;
const V_MAX_U: u32 = 0x14;
const V_ADD_F: u32 = 0x03;
/// Inline constants 0 and -1, the identities of the reductions.
const ZERO: u32 = 0x80;
const ALL_ONES: u32 = 0xC1;

/// v_lshrrev_b32 v6, 1, v0; v_mov_b32 v8, 1; v_lshlrev_b32 v7, v6, v8: v7 = 1 << (t >> 1),
/// so lanes 0..31 hold bits 0..15 and lanes 32..63 bits 16..31.
const V7_BIT: [u32; 3] = [vop2(0x16, 6, 0x81, 0), vop1(0x01, 8, 0x81), vop2(0x1A, 7, 0x106, 8)];
/// v_not_b32 v7, v7
const V7_NOT: u32 = vop1(0x37, 7, 0x107);
/// v_add_nc_u32 v7, 20, v0; v_and_b32 v7, 63, v7: v7 = (t + 20) & 63, so lanes 0..31 hold
/// 20..51 and lanes 32..63 hold 52..63 and 0..19.
const V7_ROTATED: [u32; 2] = [vop2(V_ADD_U, 7, 0x94, 0), vop2(V_AND, 7, 0xBF, 7)];
/// The section's saved EXEC, s[16:17] (`IF_VCC` saves to s[12:13]).
const SAVE_ALL: u32 = 0xBE90_25C1; // s_or_saveexec_b64 s[16:17], -1
const RESTORE: u32 = 0xBEFE_0410; // s_mov_b64 exec, s[16:17]
/// v_mov_b32 v1, s7
const V1_S7: u32 = 0x7E02_0207;
/// v_mul_u32_u24 v7, 0xabcdef, v0; v_cvt_f32_u32 v7, v7: thread t's float of t * 0xabcdef,
/// large enough that sums round.
const V7_FLOAT: [u32; 3] = [vop2(0x0B, 7, 0xFF, 0), 0x00AB_CDEF, vop1(0x06, 7, 0x107)];

/// `WaveActive<op>(v7)` into s7 as the console compiler expands it (the fxceless line's
/// `WAVE_OR_V7_TO_S7` with `op` and its identity): every lane takes part, those outside the
/// saved EXEC with `ident`; pairs, quads and octets by dpp8, rows by `row_mirror`, the two
/// rows of each 32 lanes by `v_permlanex16`, then lane 0 adds lane 32's.
const fn reduce(op: u32, ident: u32) -> [u32; 20] {
    [
        SAVE_ALL,
        0xD501_0001,
        0x0042_0E00 | ident, // v_cndmask_b32 v1, ident, v7, s[16:17]
        vop2(op, 1, 0xE9, 1),
        0xDE54_C101, // dpp8:[1,0,3,2,5,4,7,6]
        vop2(op, 1, 0xE9, 1),
        0x9360_1201, // dpp8:[2,2,0,0,6,6,4,4]
        vop2(op, 1, 0xE9, 1),
        0x0009_2401, // dpp8:[4,4,4,4,0,0,0,0]
        vop2(op, 1, 0xFA, 1),
        0xFF09_4001, // row_mirror bound_ctrl:1
        0xD778_0002,
        0x0201_0101, // v_permlanex16_b32 v2, v1, 0, 0
        vop2(op, 1, 0x102, 1),
        readlane_v1(7, 32)[0],
        readlane_v1(7, 32)[1],
        vop2(op, 1, 7, 1),
        readlane_v1(7, 0)[0],
        readlane_v1(7, 0)[1],
        RESTORE,
    ]
}

/// `WavePrefixSum(v7)` (inclusive) into v1 over the 64 lanes: in each row by `row_shr` 1, 2,
/// 4 and 8, then rows 1 and 3 add the total of the row below, then lanes 32..63 the total
/// of lanes 0..31, EXEC chosen for each step. `exclusive`: less each lane's own term.
fn prefix_sum(exclusive: bool) -> Vec<u32> {
    prefix_of(V_ADD_U, exclusive)
}

/// [`prefix_sum`] with the addition `op`.
fn prefix_of(op: u32, exclusive: bool) -> Vec<u32> {
    let mut w = vec![
        SAVE_ALL,
        0xD501_0003,
        0x0042_0E80,          // v_cndmask_b32 v3, 0, v7, s[16:17]
        vop1(0x01, 1, 0x103), // v_mov_b32 v1, v3
    ];
    for n in [1, 2, 4, 8] {
        w.extend(dpp(op, 1, 1, 0x110 + n));
    }
    w.extend(readlane_v1(14, 15));
    w.extend(readlane_v1(15, 47));
    w.extend([
        0xBEFE_03FF,
        0xFFFF_0000, // s_mov_b32 exec_lo, 0xffff0000
        0xBEFF_0380, // s_mov_b32 exec_hi, 0
        vop2(op, 1, 14, 1),
        0xBEFE_0380, // s_mov_b32 exec_lo, 0
        0xBEFF_03FF,
        0xFFFF_0000, // s_mov_b32 exec_hi, 0xffff0000
        vop2(op, 1, 15, 1),
    ]);
    w.extend(readlane_v1(14, 31));
    w.extend([
        0xBEFF_03C1, // s_mov_b32 exec_hi, -1
        vop2(op, 1, 14, 1),
    ]);
    if exclusive {
        w.extend([0xBEFE_04C1, vop2(0x26, 1, 0x101, 3)]); // s_mov_b64 exec, -1; v1 -= v3
    }
    w.push(RESTORE);
    w
}

/// A compute program of `threads` threads (thread index in v0; console waves of 64 when
/// `wave64`), 4 root constants in s0..s3 and a raw UAV's descriptor in s[8:11], then `body`.
/// `lds`: 512 bytes of LDS.
fn native(body: &[u32], wave64: bool, threads: u32, lds: bool) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    let root = |root, sgprs, user_data, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    };
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
        layout: vec![root(RootType::Constants, 4, 0, 0), root(RootType::Table, 1, 4, 1)],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            // 5 user SGPRs (s0..s4), and 512 bytes of LDS if asked for.
            rsrc2: u32::from(lds) << 15 | 5 << 1,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// One console wave of 64 running `body`, then storing v1 to its thread's dword.
fn wave(body: &[u32]) -> NativeProgram {
    native(&[body, &STORE_V1[..]].concat(), true, 64, false)
}

/// Every stage of the translation, for a host whose waves have at most `host_lanes`.
fn translate(native: &NativeProgram, host_lanes: u32) -> Result<HostShader, ShaderError> {
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, native)?;
    let container = crate::dxbc::build(&[]);
    let c = Container::parse(&container)?;
    super::emit(&c, native, &p, &st, &io, &res, host_lanes)
}

#[test]
fn encodings_decode_as_named() {
    // Decoded as a program, which ends with s_endpgm; its name is left out.
    let names = |words: &[u32]| -> Vec<&'static str> {
        let code: Vec<u8> = words.iter().chain([&S_ENDPGM]).flat_map(|w| w.to_le_bytes()).collect();
        let decoded = crate::gfx10::decode(&code).unwrap_or_else(|e| panic!("{words:#x?}: {e}"));
        let mut names: Vec<_> = decoded.iter().map(|i| i.name().unwrap()).collect();
        assert_eq!(names.pop(), Some("s_endpgm"));
        names
    };
    assert_eq!(names(&MBCNT), ["v_mbcnt_lo_u32_b32", "v_mbcnt_hi_u32_b32"]);
    assert_eq!(names(&[VCC_FROM_40]), ["v_cmp_le_u32"]);
    assert_eq!(names(&[V1_VCC_HI, V1_S12]), ["v_mov_b32", "v_mov_b32"]);
    assert_eq!(names(&[BCNT_VCC]), ["s_bcnt1_i32_b64"]);
    assert_eq!(names(&READLANE_40), ["v_readlane_b32"]);
    assert_eq!(names(&[READFIRSTLANE]), ["v_readfirstlane_b32"]);
    assert_eq!(names(&WRITELANE_40), ["v_mov_b32", "s_mov_b32", "v_writelane_b32"]);
    assert_eq!(names(&STORE_V1), ["v_lshlrev_b32", "buffer_store_dword"]);
    assert_eq!(names(&LOAD_V1), ["buffer_load_dword"]);
    assert_eq!(
        sp3(&SCALAR_LOOP_BY_WAVE)[1..],
        [
            "v_readfirstlane_b32 s16, v0",
            "s_lshr_b32 s16, s16, 6",
            "s_add_i32 s16, s16, 1",
            "v_readlane_b32 s14, v0, 10",
            "v_add_nc_u32 v1, s14, v1",
            "s_sub_i32 s16, s16, 1",
            "s_cmp_lg_u32 s16, 0",
            "s_cbranch_scc1 label_0010",
        ]
    );
    assert_eq!(
        sp3(&NESTED_SCALAR_LOOPS)[1..],
        [
            "v_readfirstlane_b32 s17, v0",
            "s_lshr_b32 s17, s17, 6",
            "s_add_i32 s17, s17, 1",
            "s_mov_b32 s16, s17",
            "v_readlane_b32 s14, v0, 10",
            "v_add_nc_u32 v1, s14, v1",
            "s_sub_i32 s16, s16, 1",
            "s_cmp_lg_u32 s16, 0",
            "s_cbranch_scc1 label_0014",
            "s_sub_i32 s17, s17, 1",
            "s_cmp_lg_u32 s17, 0",
            "s_cbranch_scc1 label_0010",
        ]
    );
    assert_eq!(
        sp3(&IF_ELSE_ONE_QUIET_WAY)[1..],
        [
            "v_readfirstlane_b32 s14, v0",
            "s_cmp_lt_u32 s14, 64",
            "s_cbranch_scc1 label_0020",
            "v_readlane_b32 s14, v0, 10",
            "v_mov_b32 v1, s14",
            "s_branch label_0024",
            "v_add_nc_u32 v1, 7, v1",
        ]
    );
    assert_eq!(sp3(&IF_ELSE_IN_AN_IF)[1..3], ["s_cmp_lg_u32 s0, 0", "s_cbranch_scc1 label_002C"]);
    assert_eq!(
        sp3(&IF_ELSE_SHARING_A_BLOCK)[3..],
        [
            "s_cbranch_scc1 label_0020",
            "v_readlane_b32 s14, v0, 10",
            "v_mov_b32 v1, s14",
            "s_branch label_002C",
            "s_cmp_lg_u32 s0, 0",
            "s_cbranch_scc1 label_0030",
            "v_add_nc_u32 v1, 7, v1",
            "v_add_nc_u32 v1, 64, v1",
        ]
    );
    assert_eq!(
        sp3(&LOOP_TWO_EXITS_BY_WAVE)[1..],
        [
            "v_readfirstlane_b32 s16, v0",
            "s_lshr_b32 s16, s16, 6",
            "s_mov_b32 s17, s16",
            "s_add_i32 s16, s16, 1",
            "v_readlane_b32 s14, v0, 10",
            "v_add_nc_u32 v1, s14, v1",
            "s_cmp_eq_u32 s17, 0",
            "s_cbranch_scc1 label_0038",
            "s_sub_i32 s16, s16, 1",
            "s_cmp_lg_u32 s16, 0",
            "s_cbranch_scc1 label_0014",
            "s_branch label_003C",
            "v_add_nc_u32 v1, 64, v1",
        ]
    );
    assert_eq!(names(&[vop2(0x1D, 3, 0x81, 0)]), ["v_xor_b32"]);
    // (the branch skips the one dword after it)
    assert_eq!(
        names(&[&IF_VCC[..], &[BARRIER, END_IF]].concat()),
        ["s_and_saveexec_b64", "s_cbranch_execz", "s_barrier", "s_mov_b64"]
    );
    assert_eq!(names(&[S14_0, S14_5]), ["s_mov_b32", "s_mov_b32"]);
    assert_eq!(
        names(&IF_FROM_40_V1_7),
        [
            "v_cmp_le_u32",
            "s_and_saveexec_b64",
            "s_cbranch_execz",
            "s_mov_b32",
            "v_mov_b32",
            "s_mov_b64"
        ]
    );
    assert_eq!(names(&DS_WRITE), ["ds_write_b32"]);
    assert_eq!(names(&LDS_WRITE_T), ["v_lshlrev_b32", "v_mov_b32", "ds_write_b32"]);
    assert_eq!(names(&LDS_READ_MIRROR), ["v_sub_nc_u32", "v_lshlrev_b32", "ds_read_b32"]);
}

/// The translation carries the used-resource list of the program it translated, which
/// the host binds descriptor tables from.
#[test]
fn a_translation_carries_its_programs_used_resources() {
    let cs = wave(&[]);
    assert!(!cs.used.is_empty());
    assert_eq!(translate(&cs, 64).unwrap().used, cs.used);
}

/// A program that reads other lanes runs in halves on a 32-lane host, and keeps its
/// console wave size on a host whose waves hold 64 lanes.
#[test]
fn a_cross_lane_program_runs_in_halves_only_on_a_32_lane_host() {
    let cs = wave(&MBCNT);
    assert_eq!(translate(&cs, 32).unwrap().wave_size, Some(32));
    assert_eq!(translate(&cs, 64).unwrap().wave_size, Some(64));
    // A console wave32 program is the host's already.
    let w32 = native(&[&MBCNT[..], &STORE_V1[..]].concat(), false, 32, false);
    assert_eq!(translate(&w32, 32).unwrap().wave_size, Some(32));
}

/// s14 = 0; s14 = 5 for the lanes from 40 on; v1 = s14 after the branch.
fn scalar_after_a_branch() -> NativeProgram {
    wave(&[&MBCNT[..], &[S14_0, VCC_FROM_40], &IF_VCC[..], &[S14_5, END_IF, V1_S14]].concat())
}

/// Where the two halves could part, a split program could hang at a group barrier or
/// keep two different copies of a scalar. The console's wave branches as one: taken
/// wave-wide, the branch keeps both halves together, and the program splits.
#[test]
fn a_branch_the_halves_could_part_at_is_taken_wave_wide() {
    let branch = |inside: u32| {
        wave(&[&MBCNT[..], &[S14_0, VCC_FROM_40], &IF_VCC[..], &[inside, END_IF]].concat())
    };
    // A group barrier only the lanes from 40 on reach.
    assert_eq!(translate(&branch(BARRIER), 32).unwrap().wave_size, Some(32));
    // A scalar written where only some lanes run and read after the branch: the half that
    // skipped it would keep the old value, where the console's whole wave has the new one.
    assert_eq!(translate(&scalar_after_a_branch(), 32).unwrap().wave_size, Some(32));
    // The same branch around per-lane code only splits.
    assert_eq!(translate(&branch(0x7E02_0280), 32).unwrap().wave_size, Some(32));
}

/// A scalar written where only some lanes run and read only there (a descriptor loaded in
/// the branch that samples with it, in the title's programs): a half reaching a read went
/// through the write, so the program splits.
#[test]
fn a_scalar_read_only_where_it_is_written_splits() {
    let cs = wave(&[&MBCNT[..], &IF_FROM_40_V1_7[..]].concat());
    assert_eq!(translate(&cs, 32).unwrap().wave_size, Some(32));
}

/// A thread group of whole console waves splits, one wave or several; a group with part of
/// a wave keeps 64-lane waves.
#[test]
fn groups_of_whole_console_waves_split() {
    let body = [&MBCNT[..], &STORE_V1[..]].concat();
    assert_eq!(translate(&native(&body, true, 128, false), 32).unwrap().wave_size, Some(32));
    assert_eq!(translate(&native(&body, true, 96, false), 32).unwrap().wave_size, Some(64));
}

/// Two console waves of 64 in one group running `body`, then storing v1 to their thread's
/// dword.
fn two_waves(body: &[u32]) -> NativeProgram {
    native(&[body, &STORE_V1[..]].concat(), true, 128, false)
}

/// v1 = v0; s16 = the wave's first thread / 64 + 1 (read from v0); then s16 rounds of
/// v1 += lane 10's v0 (read into s14): the first console wave goes round once, the
/// second twice. A loop on a scalar the console waves of a group take apart.
const SCALAR_LOOP_BY_WAVE: [u32; 10] = [
    V1_V0,
    0x7E20_0500, // v_readfirstlane_b32 s16, v0
    0x9010_8610, // s_lshr_b32 s16, s16, 6
    0x8110_8110, // s_add_i32 s16, s16, 1
    READLANE_10[0],
    READLANE_10[1],
    vop2(0x25, 1, 14, 1), // v_add_nc_u32 v1, s14, v1
    0x8190_8110,          // s_sub_i32 s16, s16, 1
    0xBF07_8010,          // s_cmp_lg_u32 s16, 0
    0xBF85_FFFA,          // s_cbranch_scc1 back to the v_readlane
];

/// [`SCALAR_LOOP_BY_WAVE`] whose rounds read lane 10 only in the lanes from 104 on (the
/// second console wave's upper lanes): a group-wide branch inside a group-wide loop.
const BRANCH_IN_A_SCALAR_LOOP: [u32; 15] = [
    V1_V0,
    0x7E20_0500, // v_readfirstlane_b32 s16, v0
    0x9010_8610, // s_lshr_b32 s16, s16, 6
    0x8110_8110, // s_add_i32 s16, s16, 1
    VCC_FROM_104[0],
    VCC_FROM_104[1],
    IF_VCC[0],
    0xBF88_0003, // s_cbranch_execz over the read and the add
    READLANE_10[0],
    READLANE_10[1],
    vop2(0x25, 1, 14, 1), // v_add_nc_u32 v1, s14, v1
    END_IF,
    0x8190_8110, // s_sub_i32 s16, s16, 1
    0xBF07_8010, // s_cmp_lg_u32 s16, 0
    0xBF85_FFF5, // s_cbranch_scc1 back to the v_cmp
];

/// v1 = v0; s17 = the wave's first thread / 64 + 1 (read from v0); then s17 rounds of a
/// loop that copies s17 into s16 and runs s16 rounds of v1 += lane 10's v0 (read into
/// s14): the first console wave goes round the inner loop once, the second 2 + 1 times.
/// A loop the console waves of a group take apart inside another.
const NESTED_SCALAR_LOOPS: [u32; 14] = [
    V1_V0,
    0x7E22_0500, // v_readfirstlane_b32 s17, v0
    0x9011_8611, // s_lshr_b32 s17, s17, 6
    0x8111_8111, // s_add_i32 s17, s17, 1
    0xBE90_0311, // s_mov_b32 s16, s17
    READLANE_10[0],
    READLANE_10[1],
    vop2(0x25, 1, 14, 1), // v_add_nc_u32 v1, s14, v1
    0x8190_8110,          // s_sub_i32 s16, s16, 1
    0xBF07_8010,          // s_cmp_lg_u32 s16, 0
    0xBF85_FFFA,          // s_cbranch_scc1 back to the v_readlane
    0x8191_8111,          // s_sub_i32 s17, s17, 1
    0xBF07_8011,          // s_cmp_lg_u32 s17, 0
    0xBF85_FFF6,          // s_cbranch_scc1 back to the s_mov_b32
];

/// A loop the whole group takes inside another is taken so too: each has its own vote.
#[test]
fn a_group_wide_loop_inside_another_splits() {
    let cs = two_waves(&NESTED_SCALAR_LOOPS);
    assert_eq!(translate(&cs, 32).unwrap().wave_size, Some(32));
    assert_eq!(lowered(&cs, 32).group_votes, 2);
}

/// A branch inside a loop goes back to itself only round the loop: it is no loop of its
/// own, but a branch the group takes in each round.
#[test]
fn a_group_wide_branch_inside_a_group_wide_loop_splits() {
    let cs = two_waves(&BRANCH_IN_A_SCALAR_LOOP);
    assert_eq!(translate(&cs, 32).unwrap().wave_size, Some(32));
}

/// A loop of one exit around a place where the halves meet, which the console waves of a
/// group leave apart, is taken by the whole group: it goes round while some console wave
/// would, a wave that left keeping out of the rounds after. A lane loop and a loop on a
/// scalar the waves take apart split.
#[test]
fn a_loop_the_waves_of_a_group_leave_apart_splits() {
    let cs = two_waves(&SCALAR_LOOP_BY_WAVE);
    assert_eq!(translate(&cs, 32).unwrap().wave_size, Some(32));
    // The loop's one branch, a loop of a single block, is the group's vote.
    assert_eq!(lowered(&cs, 32).group_votes, 1);
    assert_eq!(translate(&two_waves(&LANE_LOOP), 32).unwrap().wave_size, Some(32));
}

/// v1 = v0; s16 = the wave's first thread / 64 (its console wave) + 1 rounds, s17 = the
/// console wave. Each round, v1 += lane 10's v0 (read into s14); then the first console wave
/// leaves (s17 = 0), through a block adding 64 to v1 on its way to the loop's join; the
/// others count s16 down and leave at the loop's end once it reaches 0, through a block that
/// only jumps to the join. A loop of two exits the console waves of a group leave apart.
const LOOP_TWO_EXITS_BY_WAVE: [u32; 15] = [
    V1_V0,
    0x7E20_0500, // v_readfirstlane_b32 s16, v0
    0x9010_8610, // s_lshr_b32 s16, s16, 6
    0xBE91_0310, // s_mov_b32 s17, s16
    0x8110_8110, // s_add_i32 s16, s16, 1
    READLANE_10[0],
    READLANE_10[1],
    vop2(0x25, 1, 14, 1),   // v_add_nc_u32 v1, s14, v1
    0xBF06_8011,            // s_cmp_eq_u32 s17, 0
    0xBF85_0004,            // s_cbranch_scc1 to the add of 64
    0x8190_8110,            // s_sub_i32 s16, s16, 1
    0xBF07_8010,            // s_cmp_lg_u32 s16, 0
    0xBF85_FFF8,            // s_cbranch_scc1 back to the v_readlane
    0xBF82_0001,            // s_branch past the add of 64
    vop2(0x25, 1, 0xC0, 1), // v_add_nc_u32 v1, 64, v1
];

/// A loop of several exits around a place where the halves meet, which the console waves of
/// a group leave apart, splits: the group goes round while some console wave would, a wave
/// runs the block past its own exit in the round it leaves, and each takes, at the loop's
/// join, what it had there. A lane loop of two exits splits too.
#[test]
fn a_loop_of_several_exits_the_waves_leave_apart_splits() {
    let cs = two_waves(&LOOP_TWO_EXITS_BY_WAVE);
    assert_eq!(translate(&cs, 32).unwrap().wave_size, Some(32));
    assert_eq!(translate(&two_waves(&LOOP_TWO_EXITS), 32).unwrap().wave_size, Some(32));
}

/// In a group of several console waves, what the group cannot take as one keeps 64-lane
/// waves: a loop one exit of which ends the program, the other going on, and a scalar
/// written in a loop the group takes and read past it (a wave that left the loop early ran
/// the later rounds' scalar instructions too).
#[test]
fn what_the_waves_of_a_group_take_apart_keeps_64_lane_waves() {
    // [`LOOP_TWO_EXITS_BY_WAVE`] whose first exit ends the program.
    let returns = [&LOOP_TWO_EXITS_BY_WAVE[..14], &[S_ENDPGM]].concat();
    assert_eq!(translate(&two_waves(&returns), 32).unwrap().wave_size, Some(64));
    // v1 = s14, lane 10's v0 as the loop's last round read it.
    let read_past = [&SCALAR_LOOP_BY_WAVE[..], &[V1_S14]].concat();
    assert_eq!(translate(&two_waves(&read_past), 32).unwrap().wave_size, Some(64));
    // [`IF_ELSE_ONE_QUIET_WAY`] whose other way reads lane 40 (into s12) too: both ways meet.
    let both_meet = [
        &IF_ELSE_ONE_QUIET_WAY[..7],
        &[0xBF82_0003], // s_branch past the other way
        &READLANE_40[..],
        &[V1_S12],
    ]
    .concat();
    assert_eq!(translate(&two_waves(&both_meet), 32).unwrap().wave_size, Some(64));
}

/// v1 = the console lane (which asks for 64-lane waves); s14 = 0; s14 = 5 where some lane
/// from 104 on (the second console wave's) runs; v1 = s14 after the branch: a scalar a
/// group-wide branch joins.
const SCALAR_JOINED_PAST_104: [u32; 12] = [
    MBCNT[0],
    MBCNT[1],
    MBCNT[2],
    MBCNT[3],
    S14_0,
    VCC_FROM_104[0],
    VCC_FROM_104[1],
    IF_VCC[0],
    IF_VCC[1],
    S14_5,
    END_IF,
    V1_S14,
];

/// A scalar the join of a group-wide branch takes from inside it splits: each console wave
/// takes its own way's value there, by its own decision.
#[test]
fn a_scalar_joined_past_a_group_wide_branch_splits() {
    let cs = two_waves(&SCALAR_JOINED_PAST_104);
    assert_eq!(translate(&cs, 32).unwrap().wave_size, Some(32));
}

/// v1 = v0; a scalar branch the console waves decide each for themselves (on the wave's
/// first thread): the second wave sets v1 to its lane 10 (read into s14), a way where the
/// halves meet, and the first adds 7 to v1, a way where they do not.
const IF_ELSE_ONE_QUIET_WAY: [u32; 9] = [
    V1_V0,
    READFIRSTLANE_V0_S14,
    0xBF0A_C00E, // s_cmp_lt_u32 s14, 64: the first console wave
    0xBF85_0004, // s_cbranch_scc1 to the add
    READLANE_10[0],
    READLANE_10[1],
    V1_S14,
    0xBF82_0001,            // s_branch past the add
    vop2(0x25, 1, 0x87, 1), // v_add_nc_u32 v1, 7, v1
];

/// An if/else the whole group takes, one of whose ways has no place where the halves meet,
/// splits: the group runs the other way, then each console wave that takes the quiet one
/// runs it on its own. So does one inside an `if` joining where it does.
#[test]
fn an_if_else_with_one_quiet_way_splits() {
    let cs = two_waves(&IF_ELSE_ONE_QUIET_WAY);
    assert_eq!(translate(&cs, 32).unwrap().wave_size, Some(32));
    let cs = two_waves(&IF_ELSE_IN_AN_IF);
    assert_eq!(translate(&cs, 32).unwrap().wave_size, Some(32));
}

/// [`IF_ELSE_ONE_QUIET_WAY`] inside an `if` on root constant s0 (0, so the program enters
/// it) that joins where the if/else does.
const IF_ELSE_IN_AN_IF: [u32; 11] = [
    V1_V0,
    0xBF07_8000, // s_cmp_lg_u32 s0, 0
    0xBF85_0008, // s_cbranch_scc1 to the join
    READFIRSTLANE_V0_S14,
    0xBF0A_C00E, // s_cmp_lt_u32 s14, 64: the first console wave
    0xBF85_0004, // s_cbranch_scc1 to the add
    READLANE_10[0],
    READLANE_10[1],
    V1_S14,
    0xBF82_0001,            // s_branch past the add
    vop2(0x25, 1, 0x87, 1), // v_add_nc_u32 v1, 7, v1
];

/// [`IF_ELSE_ONE_QUIET_WAY`] whose two ways go on through one block, which adds 64 to v1,
/// before they join; the quiet way may also go straight to the join, on root constant s0
/// (0, so it does not).
const IF_ELSE_SHARING_A_BLOCK: [u32; 12] = [
    V1_V0,
    READFIRSTLANE_V0_S14,
    0xBF0A_C00E, // s_cmp_lt_u32 s14, 64: the first console wave
    0xBF85_0004, // s_cbranch_scc1 to the quiet way
    READLANE_10[0],
    READLANE_10[1],
    V1_S14,
    0xBF82_0003,            // s_branch to the shared block
    0xBF07_8000,            // s_cmp_lg_u32 s0, 0
    0xBF85_0002,            // s_cbranch_scc1 to the join
    vop2(0x25, 1, 0x87, 1), // v_add_nc_u32 v1, 7, v1
    vop2(0x25, 1, 0xC0, 1), // v_add_nc_u32 v1, 64, v1: the shared block
];

/// An if/else the whole group takes, one of whose ways is quiet, splits where its two ways
/// share a block before they join: the group runs it past the region, each console wave
/// past the quiet way.
#[test]
fn an_if_else_whose_ways_share_a_block_splits() {
    let cs = two_waves(&IF_ELSE_SHARING_A_BLOCK);
    assert_eq!(translate(&cs, 32).unwrap().wave_size, Some(32));
}

/// v_cmp_le_u32 vcc, 104, v0: lanes 40..63 of the second console wave only.
const VCC_FROM_104: [u32; 2] = [0x7D86_00FF, 104];

/// `if (t < 104) return;` as the console compiler ends a program early: a wave with no lane
/// left in EXEC jumps to an `s_endpgm` of its own. The lanes left read lane 10, turn on every
/// lane of their wave (`s_mov_b64 exec, -1`) and set v1 to it. Only the second console wave
/// goes on.
const RETURN_BELOW_104: [u32; 11] = [
    V1_V0,
    VCC_FROM_104[0],
    VCC_FROM_104[1],
    IF_VCC[0],   // s_and_saveexec_b64 s[12:13], vcc
    0xBF88_0001, // s_cbranch_execz to the s_endpgm
    0xBF82_0001, // s_branch over it
    S_ENDPGM,
    READLANE_10[0],
    READLANE_10[1],
    0xBEFE_04C1, // s_mov_b64 exec, -1
    V1_S14,
];

/// buffer_load_dword v1, v2, s[8:11], 0 offen: v1 = dword v2 / 4 of the UAV.
const LOAD_V1: [u32; 2] = [0xE030_1000, 0x8002_0102];

/// Each thread stores its index to its dword; then the lanes from 104 on (the second
/// console wave only) read lane 10 and load their dword back, adding the two in v1.
fn load_in_a_group_wide_branch() -> NativeProgram {
    let inside = [READLANE_10[0], READLANE_10[1], LOAD_V1[0], LOAD_V1[1], vop2(0x25, 1, 14, 1)];
    two_waves(
        &[
            &[V1_V0],
            &STORE_V1[..],
            &VCC_FROM_104[..],
            &[IF_VCC[0], 0xBF88_0005],
            &inside[..],
            &[END_IF],
        ]
        .concat(),
    )
}

/// A memory access in a group-wide branch runs only in a console wave that would enter
/// the branch: a wave that would have skipped it, whose scalars there may hold anything
/// (an index past the descriptor heap), makes no handle and reads nothing.
#[test]
fn a_memory_access_in_a_group_wide_branch_waits_for_the_waves_decision() {
    assert_eq!(lowered(&load_in_a_group_wide_branch(), 32).flagged_accesses, 1);
    // Outside any group-wide branch, none.
    assert_eq!(lowered(&two_waves(&[&[V1_V0], &STORE_V1[..]].concat()), 32).flagged_accesses, 0);
}

/// A memory access reading a lane mask as a number in a group-wide branch keeps 64-lane
/// waves: the mask's ballot, which the halves exchange, would come where only the console
/// waves entering run.
#[test]
fn a_memory_access_reading_a_lane_mask_in_a_group_wide_branch_keeps_64_lane_waves() {
    // buffer_load_dword v1, v2, s[8:11], vcc_lo offen
    let load_at_vcc = [LOAD_V1[0], 0x6A02_0102];
    let inside = [READLANE_10[0], READLANE_10[1], load_at_vcc[0], load_at_vcc[1]];
    let body = [&[V1_V0], &STORE_V1[..], &VCC_FROM_104[..], &[IF_VCC[0], 0xBF88_0004]];
    let cs = two_waves(&[&body.concat()[..], &inside[..], &[END_IF]].concat());
    assert_eq!(translate(&cs, 32).unwrap().wave_size, Some(64));
}

/// `if (t < 104) return;`, then the lanes left write LDS, meet at a group barrier of the
/// program's own and read their neighbour's dword (thread t ^ 1). (`v_mbcnt` first, so the
/// program asks for the console's lanes.)
fn barrier_after_an_early_return() -> NativeProgram {
    let xor1 = vop2(0x1D, 3, 0x81, 0); // v_xor_b32 v3, 1, v0
    let body = [&LDS_WRITE_T[..], &[BARRIER, xor1], &LDS_READ_MIRROR[1..]].concat();
    let mut words = MBCNT.to_vec();
    words.extend(&RETURN_BELOW_104[..7]);
    words.extend(body);
    native(&[&words[..], &STORE_V1[..]].concat(), true, 128, true)
}

/// A group barrier of the program's own in a group-wide branch: every console wave runs
/// the branch with the group, so every one reaches the barrier, as the console's waves
/// still running do. The program splits.
#[test]
fn a_group_barrier_in_a_group_wide_branch_splits() {
    assert_eq!(translate(&barrier_after_an_early_return(), 32).unwrap().wave_size, Some(32));
}

/// A branch one way of which only ends the program (`if (...) return;`) is an `if` without
/// an `else` to the group: a console wave that would have ended there runs the rest with the
/// group, its own decision guarding what it does, and ends with it. A way to the end that
/// does work first keeps 64-lane waves.
#[test]
fn a_branch_to_the_end_of_the_program_is_an_if_without_an_else() {
    let cs = two_waves(&RETURN_BELOW_104);
    assert_eq!(translate(&cs, 32).unwrap().wave_size, Some(32));
    // The way to the end stores v1 first (the s_branch skips the store and the s_endpgm).
    let mut work = RETURN_BELOW_104.to_vec();
    work.splice(5..7, [0xBF82_0004].into_iter().chain(STORE_V1).chain([S_ENDPGM]));
    assert_eq!(translate(&two_waves(&work), 32).unwrap().wave_size, Some(64));
}

/// LDS shared within one console wave: the console runs its accesses in program order, the
/// two halves at a group barrier before each, which every thread must reach. So the program
/// splits where every lane accesses LDS, and where only some lanes do once the branch
/// around the access is taken wave-wide.
#[test]
fn lds_of_one_console_wave_runs_in_halves_where_every_lane_reaches_it() {
    let every = [&MBCNT[..], &LDS_WRITE_T[..], &LDS_READ_MIRROR[..], &STORE_V1[..]].concat();
    assert_eq!(translate(&native(&every, true, 64, true), 32).unwrap().wave_size, Some(32));
    // ds_write_b32 only for the lanes from 40 on (the branch skips its two dwords).
    let some = [
        &MBCNT[..],
        &[STORE_V1[0], VCC_FROM_40, IF_VCC[0], 0xBF88_0002],
        &DS_WRITE[..],
        &[END_IF],
        &STORE_V1[..],
    ]
    .concat();
    assert_eq!(translate(&native(&some, true, 64, true), 32).unwrap().wave_size, Some(32));
}

/// The LDS dwords the halves add are theirs: a split program that allocates no LDS of its
/// own and still accesses LDS is refused, as it is unsplit, instead of reaching them.
#[test]
fn lds_the_program_does_not_allocate_is_refused_in_halves() {
    let cs = wave(&[&MBCNT[..], &[STORE_V1[0]], &DS_WRITE[..]].concat());
    for host_lanes in [32, 64] {
        let e = translate(&cs, host_lanes).err().map(|e| e.to_string());
        assert!(
            e.as_deref().is_some_and(|e| e.contains("LDS the program does not allocate")),
            "host of {host_lanes} lanes: {e:?}"
        );
    }
}

/// The SP3 text of `words` (a program ending with `s_endpgm`, left out).
fn sp3(words: &[u32]) -> Vec<String> {
    let code: Vec<u8> = words.iter().chain([&S_ENDPGM]).flat_map(|w| w.to_le_bytes()).collect();
    let decoded = crate::gfx10::decode(&code).unwrap_or_else(|e| panic!("{words:#x?}: {e}"));
    let mut text: Vec<String> = decoded.iter().map(|i| i.text(true)).collect();
    assert_eq!(text.pop().as_deref(), Some("s_endpgm"));
    text
}

#[test]
fn branch_encodings_decode_as_named() {
    assert_eq!(
        sp3(&LANE_LOOP),
        [
            "v_mov_b32 v1, v0",
            "s_mov_b64 s[12:13], exec",
            "v_readlane_b32 s14, v0, 10",
            "v_add_nc_u32 v1, s14, v1",
            "v_cmp_le_u32 vcc, 40, v1",
            "s_andn2_b64 exec, exec, vcc",
            "s_cbranch_execnz label_0008",
            "s_mov_b64 exec, s[12:13]",
        ]
    );
    assert_eq!(sp3(&[READFIRSTLANE_V0_S14]), ["v_readfirstlane_b32 s14, v0"]);
    assert_eq!(
        sp3(&UNIFORM_LOOP_IN_A_BRANCH),
        [
            "v_mov_b32 v1, v0",
            "v_cmp_le_u32 vcc, 20, v0",
            "s_and_saveexec_b64 s[18:19], vcc",
            "s_cbranch_execz label_0044",
            "s_mov_b32 s15, 0",
            "v_cmp_le_u32 vcc, 40, v0",
            "s_and_saveexec_b64 s[16:17], vcc",
            "s_cbranch_execz label_0034",
            "v_readlane_b32 s14, v0, 10",
            "v_add_nc_u32 v1, s14, v1",
            "s_or_b64 exec, exec, s[16:17]",
            "s_branch label_0038",
            "s_mov_b64 exec, s[16:17]",
            "s_add_i32 s15, s15, 1",
            "s_cmp_lt_u32 s15, 3",
            "s_cbranch_scc1 label_0014",
            "s_mov_b64 exec, s[18:19]",
        ]
    );
    let two = sp3(&LOOP_TWO_EXITS);
    assert_eq!(
        two[5..8],
        ["s_andn2_b64 exec, exec, vcc", "s_cbranch_execz label_0030", "v_add_nc_u32 v1, 1, v1"]
    );
    assert_eq!(two[10..], ["s_cbranch_execnz label_0008", "s_mov_b64 exec, s[12:13]"]);
    assert_eq!(
        sp3(&RETURN_BELOW_104)[2..],
        [
            "s_and_saveexec_b64 s[12:13], vcc",
            "s_cbranch_execz label_0018",
            "s_branch label_001C",
            "s_endpgm",
            "v_readlane_b32 s14, v0, 10",
            "s_mov_b64 exec, -1",
            "v_mov_b32 v1, s14",
        ]
    );
}

#[test]
fn section_encodings_decode_as_named() {
    assert_eq!(
        sp3(&reduce(V_ADD_U, ZERO)),
        [
            "s_or_saveexec_b64 s[16:17], -1",
            "v_cndmask_b32 v1, 0, v7, s[16:17]",
            "v_add_nc_u32 v1, v1, v1 dpp8:[1,0,3,2,5,4,7,6]",
            "v_add_nc_u32 v1, v1, v1 dpp8:[2,2,0,0,6,6,4,4]",
            "v_add_nc_u32 v1, v1, v1 dpp8:[4,4,4,4,0,0,0,0]",
            "v_add_nc_u32 v1, v1, v1 row_mirror bound_ctrl:1",
            "v_permlanex16_b32 v2, v1, 0, 0",
            "v_add_nc_u32 v1, v2, v1",
            "v_readlane_b32 s7, v1, 32",
            "v_add_nc_u32 v1, s7, v1",
            "v_readlane_b32 s7, v1, 0",
            "s_mov_b64 exec, s[16:17]",
        ]
    );
    assert_eq!(sp3(&reduce(V_AND, ALL_ONES))[1], "v_cndmask_b32 v1, -1, v7, s[16:17]");
    for (op, name) in [(V_OR, "v_or_b32"), (V_MIN_U, "v_min_u32"), (V_MAX_U, "v_max_u32")] {
        assert!(sp3(&reduce(op, ZERO))[2].starts_with(name), "{name}");
    }
    assert!(sp3(&reduce(V_ADD_F, ZERO))[2].starts_with("v_add_f32"));
    assert_eq!(sp3(&V7_FLOAT), ["v_mul_u32_u24 v7, lit(0x00abcdef), v0", "v_cvt_f32_u32 v7, v7"]);
    assert_eq!(
        sp3(&[&V7_BIT[..], &[V7_NOT], &V7_ROTATED[..], &[V1_S7]].concat()),
        [
            "v_lshrrev_b32 v6, 1, v0",
            "v_mov_b32 v8, 1",
            "v_lshlrev_b32 v7, v6, v8",
            "v_not_b32 v7, v7",
            "v_add_nc_u32 v7, 20, v0",
            "v_and_b32 v7, 63, v7",
            "v_mov_b32 v1, s7",
        ]
    );
    let prefix = sp3(&prefix_sum(true));
    assert_eq!(
        prefix[1..4],
        [
            "v_cndmask_b32 v3, 0, v7, s[16:17]",
            "v_mov_b32 v1, v3",
            "v_add_nc_u32 v1, v1, v1 row_shr:1 bound_ctrl:1"
        ]
    );
    assert_eq!(prefix[6], "v_add_nc_u32 v1, v1, v1 row_shr:8 bound_ctrl:1");
    assert_eq!(prefix[7..9], ["v_readlane_b32 s14, v1, 15", "v_readlane_b32 s15, v1, 47"]);
    assert_eq!(
        prefix[9..],
        [
            "s_mov_b32 exec_lo, lit(0xffff0000)",
            "s_mov_b32 exec_hi, 0",
            "v_add_nc_u32 v1, s14, v1",
            "s_mov_b32 exec_lo, 0",
            "s_mov_b32 exec_hi, lit(0xffff0000)",
            "v_add_nc_u32 v1, s15, v1",
            "v_readlane_b32 s14, v1, 31",
            "s_mov_b32 exec_hi, -1",
            "v_add_nc_u32 v1, s14, v1",
            "s_mov_b64 exec, -1",
            "v_sub_nc_u32 v1, v1, v3",
            "s_mov_b64 exec, s[16:17]",
        ]
    );
}

/// The program of one console wave whose threads each store the reduction `op` over the
/// wave of `v7` (made by `input`).
fn reduction(input: &[u32], op: u32, ident: u32) -> NativeProgram {
    wave(&[input, &reduce(op, ident)[..], &[V1_S7]].concat())
}

/// A whole-wave section where every lane runs splits: each half reduces its lanes and the
/// halves exchange their results. On a host of 64-lane waves nothing changes.
#[test]
fn a_wave_reduction_where_every_lane_runs_splits() {
    for (op, ident) in
        [(V_OR, ZERO), (V_AND, ALL_ONES), (V_ADD_U, ZERO), (V_MIN_U, ALL_ONES), (V_MAX_U, ZERO)]
    {
        let cs = reduction(&V7_ROTATED, op, ident);
        assert_eq!(translate(&cs, 32).unwrap().wave_size, Some(32), "op {op:#x}");
        assert_eq!(translate(&cs, 64).unwrap().wave_size, Some(64), "op {op:#x}");
    }
    for exclusive in [false, true] {
        let cs = wave(&[&V7_ROTATED[..], &prefix_sum(exclusive)].concat());
        assert_eq!(translate(&cs, 32).unwrap().wave_size, Some(32), "exclusive {exclusive}");
    }
}

/// A whole-wave section only some lanes reach: a half none of whose lanes is there would
/// not come to the exchange, so the branch around it is taken wave-wide.
#[test]
fn a_wave_reduction_in_a_lane_branch_splits() {
    assert_eq!(translate(&sum_in_a_branch(), 32).unwrap().wave_size, Some(32));
}

/// The lanes from 40 on store the sum of their (rotated) v7 over themselves; the others
/// store nothing.
fn sum_in_a_branch() -> NativeProgram {
    let body = [&reduce(V_ADD_U, ZERO)[..], &[V1_S7], &STORE_V1[..]].concat();
    let skip = 0xBF88_0000 | body.len() as u32; // s_cbranch_execz past the body
    let words = [&V7_ROTATED[..], &[VCC_FROM_40, IF_VCC[0], skip], &body, &[END_IF]].concat();
    native(&words, true, 64, false)
}

/// A lane read in a loop each lane leaves on its own: the halves could go round a different
/// number of times, so the loop is taken as the console takes it, round again while some
/// lane of the wave is left.
#[test]
fn a_lane_loop_is_taken_wave_wide() {
    assert_eq!(translate(&wave(&LANE_LOOP), 32).unwrap().wave_size, Some(32));
}

/// A loop on a scalar inside branches taken wave-wide: EXEC entering it is a guard, so the
/// join of EXEC at its header stays a join (not a constant) and its value over the back
/// edge, the latch's join, must be known too.
#[test]
fn a_uniform_loop_inside_wave_wide_branches_splits() {
    let cs = wave(&UNIFORM_LOOP_IN_A_BRANCH);
    assert_eq!(translate(&cs, 32).unwrap().wave_size, Some(32));
}

/// A loop with two exits: each lies in the other's region, so neither is outermost. Both
/// close the same loop and the console takes each for the whole wave, so they are taken
/// wave-wide together.
#[test]
fn the_exits_of_one_loop_are_taken_wave_wide_together() {
    assert_eq!(translate(&wave(&LOOP_TWO_EXITS), 32).unwrap().wave_size, Some(32));
}

/// Floats round at every addition, so a float sum is exact only in the console's order.
/// The compiler's reduction adds aligned blocks of lanes of equal size (a balanced tree),
/// which the halves can follow: it splits. A float total taken from a prefix scan adds in
/// another order and keeps 64-lane waves on two host waves; a group of at most 32 threads
/// is one host wave, which sums as an unsplit program does.
#[test]
fn a_float_wave_sum_splits_where_its_order_is_the_consoles_tree() {
    let tree = [&V7_FLOAT[..], &reduce(V_ADD_F, ZERO)[..], &[V1_S7], &STORE_V1[..]].concat();
    assert_eq!(translate(&native(&tree, true, 64, false), 32).unwrap().wave_size, Some(32));
    let scan = scan_total();
    assert_eq!(translate(&native(&scan, true, 64, false), 32).unwrap().wave_size, Some(64));
    assert_eq!(translate(&native(&scan, true, 32, false), 32).unwrap().wave_size, Some(32));
}

/// Each half adds its lanes in the tree itself, reading the partner lanes, rather than
/// trusting the host's wave sum to that order. (The card test below cannot tell the two
/// apart: the RTX 5070's 32-lane wave sum happened to add in the same tree.)
#[test]
fn a_float_tree_sum_in_halves_adds_its_lanes_itself() {
    let tree = [&V7_FLOAT[..], &reduce(V_ADD_F, ZERO)[..], &[V1_S7], &STORE_V1[..]].concat();
    let reads = |lanes| lowered(&native(&tree, true, 64, false), lanes).lane_reads;
    assert_eq!(reads(32), 5, "one read per level of the tree in a half");
    assert_eq!(reads(64), 0);
}

/// The float total of v7 read from lane 63 of an inclusive prefix scan, stored by every
/// thread.
fn scan_total() -> Vec<u32> {
    let mut scan = prefix_of(V_ADD_F, false);
    scan.pop(); // the restore, after the read
    scan.extend(readlane_v1(7, 63));
    [&V7_FLOAT[..], &scan, &[RESTORE, V1_S7], &STORE_V1[..]].concat()
}

/// [`translate`]'s lowering, for what only the lowering counts.
fn lowered(native: &NativeProgram, host_lanes: u32) -> super::lower::Lowered {
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, native).unwrap();
    super::lower::lower(&super::lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        host_lanes,
    })
    .unwrap()
}

/// A split program waits for both halves before each access to its LDS: one group barrier
/// per access, the order the console's single wave keeps by itself. (The card test below
/// cannot show that the barrier is needed: two waves of one group did not race there.)
#[test]
fn each_lds_access_of_a_split_program_waits_for_both_halves() {
    let without = native(&[&MBCNT[..], &STORE_V1[..]].concat(), true, 64, true);
    let body = [&MBCNT[..], &LDS_WRITE_T[..], &LDS_READ_MIRROR[..], &STORE_V1[..]].concat();
    let with = native(&body, true, 64, true);
    let barriers = |cs: &NativeProgram, lanes| lowered(cs, lanes).barriers;
    assert_eq!(barriers(&with, 32) - barriers(&without, 32), 2);
    assert_eq!(barriers(&with, 64), barriers(&without, 64));
}

/// Running the split programs on the host GPU, which must take waves of 32.
#[cfg(windows)]
mod card {
    use super::super::host::{Dispatch, Host, View};
    use super::super::validator;
    use super::*;

    /// The UAV's 64 dwords after one group ran `body` (each thread storing its v1), or None
    /// without a hardware device that takes the program.
    fn run(body: &[u32]) -> Option<Vec<u32>> {
        run_program(&wave(body))
    }

    /// v1 = v0; for the lanes from 40 on, v1 = v_readlane(v0, 10) (through s14).
    fn lane_read_in_a_branch() -> NativeProgram {
        let body = [READLANE_10[0], READLANE_10[1], V1_S14];
        wave(&[&[V1_V0, VCC_FROM_40, IF_VCC[0], 0xBF88_0003], &body[..], &[END_IF]].concat())
    }

    /// v1 = v_readfirstlane(v0) where EXEC was narrowed, without a branch, to the lanes
    /// from `from` on (`v_mbcnt` asks for the console's lanes, which `v_readfirstlane`
    /// alone does not).
    fn first_lane_from(from: u32) -> NativeProgram {
        // v_cmp_le_u32 vcc, from, v0
        let cmp = VCC_FROM_40 & !0xFF | (128 + from);
        wave(&[&MBCNT[..], &[cmp, IF_VCC[0], READFIRSTLANE_V0_S14, END_IF, V1_S14]].concat())
    }

    /// [`run`] for a whole program of one console wave.
    fn run_program(cs: &NativeProgram) -> Option<Vec<u32>> {
        run_group(cs, 64)
    }

    /// [`run_program`] for one group of `threads` threads: the UAV's first `threads` dwords.
    fn run_group(cs: &NativeProgram, threads: u32) -> Option<Vec<u32>> {
        let shader = translate(cs, 32).unwrap();
        assert_eq!(shader.wave_size, Some(32), "not split");
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut shader.dxil.clone()).unwrap();
        } else {
            eprintln!("no dxil.dll; validation skipped");
        }
        let Ok(host) = Host::new(false) else {
            eprintln!("no SM 6.6 hardware device; not run");
            return None;
        };
        let Ok(pipe) = host.compute_pipeline(&shader) else {
            eprintln!("the device does not take 32-lane waves; not run");
            return None;
        };
        let ud = pipe.user_data(0, |_| 0);
        let out = host
            .dispatch(
                &pipe.root,
                &pipe.pso,
                &Dispatch {
                    roots: ud.into_iter().collect(),
                    views: &[View::RawUav(threads)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        Some(out[0].clone())
    }

    #[test]
    fn each_thread_counts_its_console_lane() {
        let Some(out) = run(&MBCNT) else { return };
        assert_eq!(out, (0..64).collect::<Vec<u32>>());
    }

    #[test]
    fn a_mask_read_as_a_number_holds_both_halves() {
        if let Some(out) = run(&[VCC_FROM_40, V1_VCC_HI]) {
            assert_eq!(out, vec![0xFFFF_FF00; 64], "lanes 40..63 in the high dword");
        }
        if let Some(out) = run(&[VCC_FROM_40, BCNT_VCC, V1_S12]) {
            assert_eq!(out, vec![24; 64], "24 lanes from 40 on");
        }
    }

    #[test]
    fn a_lane_read_reaches_the_other_half() {
        let Some(out) = run(&[&READLANE_40[..], &[V1_S12]].concat()) else { return };
        assert_eq!(out, vec![40; 64]);
    }

    /// In a split program (`v_mbcnt` makes it read lanes), the first lane is console lane
    /// 0's in both halves. (`v_readfirstlane` alone does not ask for the console's lanes:
    /// such a program declares no wave size and is not split.)
    #[test]
    fn the_first_lane_is_the_console_waves() {
        let Some(out) = run(&[&MBCNT[..], &[READFIRSTLANE, V1_S12]].concat()) else { return };
        assert_eq!(out, vec![0; 64]);
    }

    #[test]
    fn a_lane_write_reaches_its_console_lane() {
        let Some(out) = run(&WRITELANE_40) else { return };
        let want: Vec<u32> = (0..64).map(|t| if t == 40 { 7 } else { 0 }).collect();
        assert_eq!(out, want);
    }

    /// Each thread reads the LDS dword the mirror thread of the other half wrote just before:
    /// in the console's one wave the write comes first.
    #[test]
    fn lds_written_by_one_half_is_read_by_the_other() {
        let body = [&MBCNT[..], &LDS_WRITE_T[..], &LDS_READ_MIRROR[..], &STORE_V1[..]].concat();
        let Some(out) = run_program(&native(&body, true, 64, true)) else { return };
        assert_eq!(out, (0..64).rev().collect::<Vec<u32>>());
    }

    /// A wave reduction is of all 64 console lanes. Each input makes at least one half's
    /// own reduction differ from the wave's.
    #[test]
    fn a_wave_reduction_covers_both_halves() {
        let rotated: u32 = (0..64).map(|t| (t + 20) & 63).sum();
        let not_bit = [&V7_BIT[..], &[V7_NOT]].concat();
        let cases = [
            ("or", reduction(&V7_BIT, V_OR, ZERO), u32::MAX),
            ("and", reduction(&not_bit, V_AND, ALL_ONES), 0),
            ("sum", reduction(&V7_ROTATED, V_ADD_U, ZERO), rotated),
            ("min", reduction(&V7_ROTATED, V_MIN_U, ALL_ONES), 0),
            ("max", reduction(&V7_ROTATED, V_MAX_U, ZERO), 63),
        ];
        for (name, cs, want) in cases {
            let Some(out) = run_program(&cs) else { return };
            assert_eq!(out, vec![want; 64], "{name}");
        }
    }

    /// A float wave sum over the two halves is the console's to the bit: the balanced tree
    /// over the lanes in their order, which a sum in lane order does not give here.
    #[test]
    fn a_float_wave_sum_is_the_consoles_tree_bit_for_bit() {
        #[allow(clippy::cast_precision_loss, reason = "the console's conversion rounds too")]
        let terms: Vec<f32> = (0..64u32).map(|t| (t * 0xAB_CDEF) as f32).collect();
        let mut tree = terms.clone();
        while tree.len() > 1 {
            tree = tree.chunks(2).map(|c| c[0] + c[1]).collect();
        }
        let in_a_row: f32 = terms.iter().sum();
        assert_ne!(tree[0].to_bits(), in_a_row.to_bits(), "these terms tell the orders apart");
        let body = [&V7_FLOAT[..], &reduce(V_ADD_F, ZERO)[..], &[V1_S7], &STORE_V1[..]].concat();
        let Some(out) = run_program(&native(&body, true, 64, false)) else { return };
        assert_eq!(out, vec![tree[0].to_bits(); 64], "tree {:#x}", tree[0].to_bits());
    }

    /// A prefix sum in the upper half goes on from the lower half's total.
    #[test]
    fn a_wave_prefix_sum_runs_on_across_the_halves() {
        let terms: Vec<u32> = (0..64).map(|t| (t + 20) & 63).collect();
        for exclusive in [false, true] {
            let Some(out) = run(&[&V7_ROTATED[..], &prefix_sum(exclusive)].concat()) else {
                return;
            };
            let want: Vec<u32> =
                (0..64).map(|t| terms[..t + usize::from(!exclusive)].iter().sum()).collect();
            assert_eq!(out, want, "exclusive {exclusive}");
        }
    }

    /// Only lanes of the upper half take the branch, and read a lane of the lower half.
    #[test]
    fn a_lane_read_in_a_lane_branch_reaches_the_other_half() {
        let Some(out) = run_program(&lane_read_in_a_branch()) else { return };
        let want: Vec<u32> = (0..64).map(|t| if t < 40 { t } else { 10 }).collect();
        assert_eq!(out, want);
    }

    /// Each lane leaves the loop on its own, and every round reads lane 10 of the lower
    /// half: the upper half's lanes go round fewer times than lane 0.
    #[test]
    fn a_lane_read_in_a_lane_loop_reaches_the_other_half() {
        let Some(out) = run_program(&wave(&LANE_LOOP)) else { return };
        let want: Vec<u32> = (0..64)
            .map(|t| {
                let mut v = t + 10;
                while v < 40 {
                    v += 10;
                }
                v
            })
            .collect();
        assert_eq!(out, want);
    }

    /// The upper half's lanes all leave at the first exit of the first round; the lower
    /// half's go on reading lane 10.
    #[test]
    fn a_lane_read_in_a_loop_of_two_exits_reaches_the_other_half() {
        let Some(out) = run_program(&wave(&LOOP_TWO_EXITS)) else { return };
        let want: Vec<u32> = (0..64)
            .map(|t| {
                let mut v = t;
                loop {
                    v += 10;
                    if v >= 40 {
                        break v;
                    }
                    v += 1;
                    if v >= 40 {
                        break v;
                    }
                }
            })
            .collect();
        assert_eq!(out, want);
    }

    /// The first console wave goes round once and adds 64 on its way out; the second goes
    /// round twice.
    #[test]
    fn a_loop_of_several_exits_runs_each_waves_way_out() {
        let Some(out) = run_group(&two_waves(&LOOP_TWO_EXITS_BY_WAVE), 128) else { return };
        let want: Vec<u32> =
            (0..128).map(|t| if t < 64 { t + 10 + 64 } else { t + 2 * 74 }).collect();
        assert_eq!(out, want);
    }

    /// [`a_lane_read_in_a_loop_of_two_exits_reaches_the_other_half`] in a group of two
    /// console waves, each adding its own lane 10: the lanes leave at either exit, the second
    /// wave's after its first round.
    #[test]
    fn a_lane_loop_of_two_exits_runs_each_lanes_rounds_in_a_group() {
        let Some(out) = run_group(&two_waves(&LOOP_TWO_EXITS), 128) else { return };
        let want: Vec<u32> = (0..128)
            .map(|t| {
                let (mut v, add) = (t, t / 64 * 64 + 10);
                loop {
                    v += add;
                    if v >= 40 {
                        break v;
                    }
                    v += 1;
                    if v >= 40 {
                        break v;
                    }
                }
            })
            .collect();
        assert_eq!(out, want);
    }

    /// The lanes from 40 on add lane 10's v0 in each of three rounds; the others keep v0.
    #[test]
    fn a_lane_read_in_a_uniform_loop_inside_a_branch_reaches_the_other_half() {
        let Some(out) = run_program(&wave(&UNIFORM_LOOP_IN_A_BRANCH)) else { return };
        let want: Vec<u32> = (0..64).map(|t| if t < 40 { t } else { t + 30 }).collect();
        assert_eq!(out, want);
    }

    /// The first lane of a narrowed EXEC, in either half.
    #[test]
    fn the_first_lane_of_a_narrowed_exec_is_the_console_waves() {
        for from in [10, 40] {
            let Some(out) = run_program(&first_lane_from(from)) else { return };
            assert_eq!(out, vec![from; 64], "EXEC from lane {from}");
        }
    }

    /// The console's wave takes the branch, so its one s14 is 5 in every lane.
    #[test]
    fn a_scalar_written_in_a_lane_branch_is_the_whole_waves() {
        let Some(out) = run_program(&scalar_after_a_branch()) else { return };
        assert_eq!(out, vec![5; 64]);
    }

    /// A wave sum in a branch is of the lanes that took it.
    #[test]
    fn a_wave_sum_in_a_lane_branch_is_of_its_lanes() {
        let Some(out) = run_program(&sum_in_a_branch()) else { return };
        let sum: u32 = (40..64).map(|t| (t + 20) & 63).sum();
        let want: Vec<u32> = (0..64).map(|t| if t < 40 { 0 } else { sum }).collect();
        assert_eq!(out, want);
    }

    /// The lanes from 40 on take a scalar written in their branch; the others keep their
    /// console lane.
    #[test]
    fn a_scalar_written_in_a_lane_branch_reaches_its_lanes() {
        let Some(out) = run(&[&MBCNT[..], &IF_FROM_40_V1_7[..]].concat()) else { return };
        let want: Vec<u32> = (0..64).map(|t| if t < 40 { t } else { 7 }).collect();
        assert_eq!(out, want);
    }

    // A group of two console waves (128 threads, four host waves) whose branch around a
    // lane read only one console wave would take. A group barrier inside it that the other
    // wave never reaches does not hang on every GPU: one whose barriers count arrivals lets
    // the waves pass at different barriers, so the check is the data, bit for bit.

    /// A lane branch only the second console wave would take, whose lanes read lane 10 of
    /// its lower half. Then every lane adds lane 40's v0, of its own wave's upper half, so
    /// the first wave meets its other half where the second may still be inside the branch.
    /// The whole group takes the branch, and the first wave, inside it, changes nothing.
    #[test]
    fn a_lane_branch_one_console_wave_takes_reads_its_own_lower_half() {
        let body = [READLANE_10[0], READLANE_10[1], V1_S14];
        let after = [READLANE_40[0], READLANE_40[1], vop2(0x25, 1, 12, 1)]; // v1 += s12
        let cs = two_waves(
            &[
                &[V1_V0],
                &VCC_FROM_104[..],
                &[IF_VCC[0], 0xBF88_0003],
                &body[..],
                &[END_IF],
                &after[..],
            ]
            .concat(),
        );
        let Some(out) = run_group(&cs, 128) else { return };
        let want: Vec<u32> =
            (0..128).map(|t| if t < 104 { t } else { 74 } + t / 64 * 64 + 40).collect();
        assert_eq!(out, want);
    }

    /// The same branch, whose lanes read lane 10 and then turn on every lane of their wave
    /// (`s_mov_b64 exec, -1`) to set v1 to 7. The first console wave, which would have
    /// skipped the branch, runs it with the group, and only its own decision keeps its lanes
    /// out: EXEC turned on inside would not.
    #[test]
    fn a_wave_that_would_skip_a_group_wide_branch_keeps_its_lanes_out() {
        let body = [READLANE_10[0], READLANE_10[1], 0xBEFE_04C1, 0x7E02_0287]; // exec = -1; v1 = 7
        let after = [READLANE_40[0], READLANE_40[1], vop2(0x25, 1, 12, 1)]; // v1 += s12
        let cs = two_waves(
            &[
                &[V1_V0],
                &VCC_FROM_104[..],
                &[IF_VCC[0], 0xBF88_0004],
                &body[..],
                &[END_IF],
                &after[..],
            ]
            .concat(),
        );
        let Some(out) = run_group(&cs, 128) else { return };
        let want: Vec<u32> = (0..128).map(|t| if t < 64 { t + 40 } else { 7 + 104 }).collect();
        assert_eq!(out, want);
    }

    /// A scalar branch the console waves decide each for themselves (on the wave's first
    /// thread): only the second reads lane 10, its own.
    fn scalar_branch_one_wave_takes() -> NativeProgram {
        two_waves(&[
            V1_V0,
            READFIRSTLANE_V0_S14,
            0xBF0A_C00E, // s_cmp_lt_u32 s14, 64: the first console wave
            0xBF85_0003, // s_cbranch_scc1 over the read
            READLANE_10[0],
            READLANE_10[1],
            V1_S14,
        ])
    }

    /// A scalar branch only the second console wave takes, around a read of its lane 10:
    /// the whole group takes it, and the first wave, inside it, changes nothing.
    #[test]
    fn a_scalar_branch_one_console_wave_takes_reads_its_own_lane() {
        let Some(out) = run_group(&scalar_branch_one_wave_takes(), 128) else { return };
        let want: Vec<u32> = (0..128).map(|t| if t < 64 { t } else { 74 }).collect();
        assert_eq!(out, want);
    }

    /// The first console wave returns early, before the second reads its lane 10 and stores
    /// from every lane: the first runs the rest with the group, and its own decision, not
    /// the EXEC turned on there, keeps it from storing.
    #[test]
    fn a_wave_that_returns_early_stores_nothing() {
        let Some(out) = run_group(&two_waves(&RETURN_BELOW_104), 128) else { return };
        let want: Vec<u32> = (0..128).map(|t| if t < 64 { 0 } else { 74 }).collect();
        assert_eq!(out, want);
    }

    /// The load in the second console wave's branch reads what its thread stored; the
    /// first wave, which skips the load, keeps its v1.
    #[test]
    fn a_load_in_a_group_wide_branch_reads_for_the_waves_that_enter() {
        let Some(out) = run_group(&load_in_a_group_wide_branch(), 128) else { return };
        let want: Vec<u32> = (0..128).map(|t| if t < 104 { t } else { t + 74 }).collect();
        assert_eq!(out, want);
    }

    /// Each lane adds its wave's lane 10 until it reaches 40: the second console wave's
    /// lanes (from 64 on) leave after the first round, the first wave's after up to four.
    /// The group goes round four times; the second wave adds nothing after its first.
    #[test]
    fn a_lane_loop_the_waves_leave_apart_runs_until_the_last_leaves() {
        let Some(out) = run_group(&two_waves(&LANE_LOOP), 128) else { return };
        let want: Vec<u32> = (0..128)
            .map(|t| {
                let mut v = t + (t / 64 * 64 + 10);
                while v < 40 {
                    v += 10;
                }
                v
            })
            .collect();
        assert_eq!(out, want);
    }

    /// The first console wave goes round once, the second twice, each adding its own
    /// lane 10.
    #[test]
    fn a_scalar_loop_each_wave_counts_runs_each_waves_rounds() {
        let Some(out) = run_group(&two_waves(&SCALAR_LOOP_BY_WAVE), 128) else { return };
        let want: Vec<u32> = (0..128).map(|t| if t < 64 { t + 10 } else { t + 2 * 74 }).collect();
        assert_eq!(out, want);
    }

    /// Each round, only the second console wave's lanes from 104 on add their lane 10;
    /// that wave goes round twice.
    #[test]
    fn a_branch_inside_a_scalar_loop_runs_in_each_waves_rounds() {
        let Some(out) = run_group(&two_waves(&BRANCH_IN_A_SCALAR_LOOP), 128) else { return };
        let want: Vec<u32> = (0..128).map(|t| if t < 104 { t } else { t + 2 * 74 }).collect();
        assert_eq!(out, want);
    }

    /// The first console wave goes round the inner loop once, the second three times (2 + 1),
    /// each adding its own lane 10.
    #[test]
    fn a_loop_inside_another_runs_each_waves_rounds() {
        let Some(out) = run_group(&two_waves(&NESTED_SCALAR_LOOPS), 128) else { return };
        let want: Vec<u32> = (0..128).map(|t| if t < 64 { t + 10 } else { t + 3 * 74 }).collect();
        assert_eq!(out, want);
    }

    /// Only the second console wave enters the branch, and only it takes s14 = 5 there; the
    /// first, which ran the branch's scalars with the group, keeps 0.
    #[test]
    fn a_scalar_joined_past_a_group_wide_branch_is_each_waves_own() {
        let Some(out) = run_group(&two_waves(&SCALAR_JOINED_PAST_104), 128) else { return };
        let want: Vec<u32> = (0..128).map(|t| if t < 64 { 0 } else { 5 }).collect();
        assert_eq!(out, want);
    }

    /// The second console wave takes the way that reads its lane 10, which the group runs;
    /// the first then adds 7 on its own way. The same inside an `if` joining there.
    #[test]
    fn an_if_else_with_one_quiet_way_runs_each_waves_way() {
        let want: Vec<u32> = (0..128).map(|t| if t < 64 { t + 7 } else { 74 }).collect();
        let Some(out) = run_group(&two_waves(&IF_ELSE_ONE_QUIET_WAY), 128) else { return };
        assert_eq!(out, want);
        let Some(out) = run_group(&two_waves(&IF_ELSE_IN_AN_IF), 128) else { return };
        assert_eq!(out, want);
    }

    /// The second console wave sets v1 to its lane 10 on the way the group runs, the first
    /// adds 7 on its own way; both then add 64 in the block their ways share.
    #[test]
    fn an_if_else_whose_ways_share_a_block_runs_it_on_each_way() {
        let want: Vec<u32> = (0..128).map(|t| if t < 64 { t + 71 } else { 138 }).collect();
        let Some(out) = run_group(&two_waves(&IF_ELSE_SHARING_A_BLOCK), 128) else { return };
        assert_eq!(out, want);
    }

    /// The first console wave returns early; the lanes left read their neighbour's LDS
    /// dword past a group barrier.
    #[test]
    fn a_group_barrier_after_an_early_return_orders_the_waves_left() {
        let Some(out) = run_group(&barrier_after_an_early_return(), 128) else { return };
        let want: Vec<u32> = (0..128).map(|t| if t < 104 { 0 } else { t ^ 1 }).collect();
        assert_eq!(out, want);
    }
}

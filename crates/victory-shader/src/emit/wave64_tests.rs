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

/// Where the two halves could part, a split program could hang at a group barrier or
/// keep two different copies of a scalar: those programs keep 64-lane waves.
#[test]
fn programs_whose_halves_could_part_keep_64_lane_waves() {
    let branch = |inside: u32| {
        wave(&[&MBCNT[..], &[S14_0, VCC_FROM_40], &IF_VCC[..], &[inside, END_IF]].concat())
    };
    // A group barrier only the lanes from 40 on reach.
    assert_eq!(translate(&branch(BARRIER), 32).unwrap().wave_size, Some(64));
    // A scalar written where only some lanes run and read after the branch: the half that
    // skipped it would keep the old value, where the console's whole wave has the new one.
    let after =
        wave(&[&MBCNT[..], &[S14_0, VCC_FROM_40], &IF_VCC[..], &[S14_5, END_IF, V1_S14]].concat());
    assert_eq!(translate(&after, 32).unwrap().wave_size, Some(64));
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

/// This first step splits one console wave per thread group, without LDS of the program's
/// own: other shapes keep 64-lane waves.
#[test]
fn other_group_shapes_keep_64_lane_waves() {
    let body = [&MBCNT[..], &STORE_V1[..]].concat();
    assert_eq!(translate(&native(&body, true, 128, false), 32).unwrap().wave_size, Some(64));
}

/// LDS shared within one console wave: the console runs its accesses in program order, the
/// two halves at a group barrier before each, which every thread must reach. So the program
/// splits where every lane accesses LDS, and keeps 64-lane waves where only some lanes do.
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
    assert_eq!(translate(&native(&some, true, 64, true), 32).unwrap().wave_size, Some(64));
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
    use super::super::host::{Dispatch, Host, Root, View};
    use super::super::validator;
    use super::*;

    /// The UAV's 64 dwords after one group ran `body` (each thread storing its v1), or None
    /// without a hardware device that takes the program.
    fn run(body: &[u32]) -> Option<Vec<u32>> {
        run_program(&wave(body))
    }

    /// [`run`] for a whole program of one console wave.
    fn run_program(cs: &NativeProgram) -> Option<Vec<u32>> {
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
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            eprintln!("the device does not take 32-lane waves; not run");
            return None;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(64)],
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

    /// The lanes from 40 on take a scalar written in their branch; the others keep their
    /// console lane.
    #[test]
    fn a_scalar_written_in_a_lane_branch_reaches_its_lanes() {
        let Some(out) = run(&[&MBCNT[..], &IF_FROM_40_V1_7[..]].concat()) else { return };
        let want: Vec<u32> = (0..64).map(|t| if t < 40 { t } else { 7 }).collect();
        assert_eq!(out, want);
    }
}

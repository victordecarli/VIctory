//! Pipeline states. Both descriptions start as on PC: the root signature,
//! then the shader stages (five for graphics, one for compute); streams
//! carry the rest of the graphics state as subobjects of the standard
//! types.
//!
//! Host pipelines are made on a pipeline's first draw, not at creation:
//! Grand Theft Auto VI creates some 60,000 pipelines in its first 20
//! seconds and draws with a few dozen. Shader containers are shared by
//! content hash and translated once, on first use.
//!
//! shortcut: only vertex plus pixel pipelines from streams and compute
//! pipelines get host pipelines; mesh, tessellation and geometry stages,
//! and the plain graphics description, keep their containers only. The 32-byte
//! packet after the header stays zero: titles only copy it inline, which
//! our null command-list limits keep from running.

use std::collections::HashMap;
use std::mem::offset_of;
use std::sync::{Arc, Mutex, OnceLock};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, PIPELINE, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const PIPELINE_IIDS: [Iid; 4] = [
    guid(0x765a30f3, 0xf624, 0x4c6f, [0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45]),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
pub(crate) struct Bytecode {
    code: *const u8,
    size: usize,
}

#[repr(C)]
pub(crate) struct GraphicsHead {
    root: This,
    /// Vertex, pixel, domain, hull, geometry.
    stages: [Bytecode; 5],
}

#[repr(C)]
pub(crate) struct ComputeHead {
    root: This,
    stage: Bytecode,
}

/// A shader container a title passed, shared by every pipeline with the
/// same bytes.
pub(crate) struct Shader {
    pub(crate) stage: &'static str,
    /// The first 8 bytes of its BLAKE3, as logged.
    pub(crate) hash: String,
    container: Vec<u8>,
    /// Shared, so a host pipeline's stages can hold it (see `host::Stage`).
    #[cfg(windows)]
    translated: OnceLock<Option<Arc<Translated>>>,
}

impl std::fmt::Debug for Shader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.stage, self.hash)
    }
}

/// Every container seen, by full content hash.
static SHADERS: Mutex<Option<HashMap<[u8; 32], Arc<Shader>>>> = Mutex::new(None);

/// The shared container for `bytes`.
fn shader(stage: &'static str, bytes: &[u8]) -> Arc<Shader> {
    let full = blake3::hash(bytes);
    let mut shaders = SHADERS.lock().unwrap_or_else(|e| e.into_inner());
    shaders
        .get_or_insert_with(HashMap::new)
        .entry(*full.as_bytes())
        .or_insert_with(|| {
            Arc::new(Shader {
                stage,
                hash: full.to_hex()[..16].to_owned(),
                container: bytes.to_vec(),
                #[cfg(windows)]
                translated: OnceLock::new(),
            })
        })
        .clone()
}

/// A translated shader and the console resources its code reads.
#[cfg(windows)]
pub(crate) struct Translated {
    pub(crate) host: victory_shader::emit::HostShader,
    pub(crate) used: Vec<victory_shader::UsedResource>,
}

#[cfg(windows)]
impl Shader {
    /// The translation, made on first use; a shader that does not
    /// translate is logged once.
    pub(crate) fn translated(&self) -> Option<&Arc<Translated>> {
        self.translated
            .get_or_init(|| {
                let done = translate(&self.container);
                match &done {
                    Ok(t) => tracing::info!(
                        stage = self.stage,
                        hash = %self.hash,
                        user_data = t.host.user_data.len(),
                        samplers = t.host.samplers.len(),
                        "shader translated"
                    ),
                    Err(e) => tracing::warn!(
                        stage = self.stage,
                        hash = %self.hash,
                        error = %e,
                        "shader does not translate; its pipelines' draws are dropped"
                    ),
                }
                done.ok().map(Arc::new)
            })
            .as_ref()
    }
}

#[cfg(windows)]
fn translate(container: &[u8]) -> Result<Translated, String> {
    use victory_shader::{Container, FourCC, HwStage};
    let host = crate::shader_cache::host_shader(container)?;
    let c = Container::parse(container).map_err(|e| e.to_string())?;
    let used = match c.part(FourCC::DXIL) {
        Some(part) => victory_shader::native_programs(part.data)
            .map_err(|e| e.to_string())?
            .into_iter()
            .find(|p| matches!(p.stage, HwStage::Vs | HwStage::Ps | HwStage::Ms | HwStage::Cs))
            .map(|p| p.used)
            .unwrap_or_default(),
        None => Vec::new(),
    };
    Ok(Translated { host, used })
}

#[repr(C)]
struct Pipeline {
    object: Object,
    packet: [u32; 8],
    /// Each used stage's container.
    shaders: Vec<Arc<Shader>>,
    /// Whether a list has bound it yet (its shaders are logged then).
    bound: std::sync::atomic::AtomicBool,
    /// What its host pipeline is made from, and the host pipeline once made
    /// (by a worker: the title may release the pipeline meanwhile).
    #[cfg(windows)]
    host: Arc<make::Pending>,
}
const _: () = assert!(offset_of!(Pipeline, packet) == 0x10);

/// Stage names, in description order.
const GRAPHICS_STAGES: [&str; 5] = ["vertex", "pixel", "domain", "hull", "geometry"];

/// # Safety
///
/// `this` is a live pipeline this module created, and it stays alive for
/// `'a`.
unsafe fn pipeline<'a>(this: This) -> &'a Pipeline {
    // SAFETY: `this` is a live pipeline, per this function's contract.
    unsafe { &*this.cast::<Pipeline>() }
}

/// List method: logs each pipeline's shaders the first time a list binds
/// it, so a run lists the shaders its frames need first.
///
/// # Safety
///
/// `this` is null or a live pipeline this module created.
pub(crate) unsafe fn bound(this: This) {
    if this.is_null() {
        return;
    }
    // SAFETY: non-null, so a live pipeline, per this function's contract.
    let p = unsafe { pipeline(this) };
    if p.bound.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let frame = crate::frame::current();
    for shader in &p.shaders {
        tracing::info!(pipeline = ?this, frame, stage = shader.stage, size = shader.container.len(), hash = %shader.hash, "first bind");
    }
}

/// The host pipeline of a title pipeline; `None` if one of its shaders does
/// not translate, the host refuses it, or it is still being made (its first
/// use starts that, see [`make`]).
#[cfg(windows)]
/// # Safety
///
/// `this` is null or a live pipeline this module created, and it stays
/// alive for `'a`.
pub(crate) unsafe fn host_pipeline<'a>(this: This) -> Option<&'a host::HostPipeline> {
    if this.is_null() {
        return None;
    }
    // SAFETY: non-null, so a live pipeline, per this function's contract.
    make::ready(&unsafe { pipeline(this) }.host)
}

/// Host pipelines queued and not made yet.
#[cfg(windows)]
pub(crate) fn making() -> u64 {
    make::waiting()
}

/// Draws and dispatches dropped since the last [`take_drops`], by reason
/// (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
static DROPS: Mutex<Option<HashMap<String, u64>>> = Mutex::new(None);

/// Shader hashes as printed in the pipeline diagnostics.
#[cfg(all(windows, feature = "diagnostics"))]
fn parse_skip_hashes(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|hash| !hash.is_empty())
        .filter_map(|hash| {
            if hash.len() == 16 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                Some(hash.to_ascii_lowercase())
            } else {
                tracing::warn!(hash, "VICTORY_SKIP_PIPELINES expects 16-digit shader hashes");
                None
            }
        })
        .collect()
}

#[cfg(all(windows, feature = "diagnostics"))]
fn skipped_shader(p: &Pipeline) -> Option<&str> {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    let hashes = HASHES.get_or_init(|| {
        let value = std::env::var("VICTORY_SKIP_PIPELINES").unwrap_or_default();
        parse_skip_hashes(&value)
    });
    if hashes.is_empty() {
        return None;
    }
    p.shaders.iter().find(|shader| hashes.contains(&shader.hash)).map(|shader| shader.hash.as_str())
}

#[cfg(all(windows, feature = "diagnostics"))]
/// # Safety
///
/// `this` is null or a live pipeline this module created.
pub(crate) unsafe fn is_skipped(this: This) -> bool {
    // SAFETY: non-null when it is read, so a live pipeline, per this
    // function's contract.
    !this.is_null() && skipped_shader(unsafe { pipeline(this) }).is_some()
}

/// Counts a draw or dispatch of `this` intentionally skipped or dropped
/// for want of a host pipeline (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
/// # Safety
///
/// `this` is null or a live pipeline this module created.
pub(crate) unsafe fn note_drop(this: This) {
    let why = if this.is_null() {
        Some("no pipeline set".to_owned())
    } else {
        // SAFETY: non-null, so a live pipeline, per this function's contract.
        let p = unsafe { pipeline(this) };
        skipped_shader(p)
            .map(|hash| format!("skipped shader {hash}"))
            .or_else(|| make::missing(&p.host))
    };
    if let Some(why) = why {
        note_why(why);
    }
}

/// Counts a draw or dispatch dropped for `why` (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_why(why: String) {
    let mut drops = DROPS.lock().unwrap_or_else(|e| e.into_inner());
    *drops.get_or_insert_with(HashMap::new).entry(why).or_default() += 1;
}

/// The drops counted since the last call, most first (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn take_drops() -> Vec<(String, u64)> {
    let drops = DROPS.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut drops: Vec<(String, u64)> = drops.into_iter().flatten().collect();
    // Frame reports only show the first 40 reasons. Keep ties ordered so
    // a HashMap iteration order change cannot change which reasons appear.
    drops.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    drops
}

/// The stage's container, if it has one.
///
/// # Safety
///
/// `code.code` is null or valid for reads of `code.size` bytes for the
/// duration of the call.
unsafe fn copy(stage: &'static str, code: &Bytecode) -> Option<Arc<Shader>> {
    if code.code.is_null() || code.size == 0 {
        return None;
    }
    // SAFETY: non-null, and `size` readable bytes, per this function's
    // contract.
    Some(shader(stage, unsafe { std::slice::from_raw_parts(code.code, code.size) }))
}

/// # Safety
///
/// `iid` is null or points to an interface id valid for reads, and `out` is
/// null or valid for a write of an object pointer, for the duration of the
/// call.
unsafe fn hand_out(
    shaders: Vec<Arc<Shader>>,
    #[cfg(windows)] graphics: Option<Box<host::GraphicsState>>,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::trace!(?shaders, "pipeline state");
    #[cfg(windows)]
    let host = Arc::new(make::Pending::new(shaders.clone(), graphics));
    #[cfg(windows)]
    make::early(&host);
    let pipeline = Box::new(Pipeline {
        object: Object::new(&FUNCTIONS),
        packet: [0; 8],
        #[cfg(windows)]
        host,
        shaders,
        bound: std::sync::atomic::AtomicBool::new(false),
    });
    object::hand_out(PIPELINE, pipeline, iid, &PIPELINE_IIDS, out)
}

/// Device method.
///
/// # Safety
///
/// `desc` is null or valid for reads of a [`GraphicsHead`], and each
/// shader's `code` is null or valid for reads of its `size` bytes. `iid` is
/// null or points to an interface id valid for reads, and `out` is null or
/// valid for a write of an object pointer, for the duration of the call.
pub(crate) unsafe extern "system" fn create_graphics(
    _: This,
    desc: *const GraphicsHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: `desc` is null or readable, per this function's contract.
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let stages = GRAPHICS_STAGES.iter().zip(&desc.stages);
    // SAFETY: each shader's code is as this function's contract gives it.
    let shaders = stages.filter_map(|(s, c)| unsafe { copy(s, c) }).collect();
    // SAFETY: `iid` and `out` as this function's contract gives them.
    unsafe {
        hand_out(
            shaders,
            #[cfg(windows)]
            None,
            iid,
            out,
        )
    }
}

/// Device method.
///
/// # Safety
///
/// `desc` is null or valid for reads of a [`ComputeHead`], and each
/// shader's `code` is null or valid for reads of its `size` bytes. `iid` is
/// null or points to an interface id valid for reads, and `out` is null or
/// valid for a write of an object pointer, for the duration of the call.
pub(crate) unsafe extern "system" fn create_compute(
    _: This,
    desc: *const ComputeHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: `desc` is null or readable, per this function's contract.
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    // SAFETY: the shader's code, `iid` and `out` are as this function's
    // contract gives them.
    unsafe {
        hand_out(
            copy("compute", &desc.stage).into_iter().collect(),
            #[cfg(windows)]
            None,
            iid,
            out,
        )
    }
}

/// Device method 0x1f0: the plain compute description plus console
/// extensions `(desc, extension count, extensions, iid, out)`.
/// shortcut: the extensions (console shader settings) are ignored, as is
/// the rest of the description.
///
/// # Safety
///
/// As [`create_compute`]; the extensions are not read.
pub(crate) unsafe extern "system" fn create_compute_x(
    this: This,
    desc: *const ComputeHead,
    _extensions: u32,
    _extended: *const u8,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the pointers are as this function's contract gives them.
    unsafe { create_compute(this, desc, iid, out) }
}

/// A pipeline described as a stream: its size and the subobjects, each a
/// type then its data, padded to 8 bytes.
#[repr(C)]
pub(crate) struct StreamDesc {
    size: usize,
    stream: *const u8,
}

/// A subobject type's size (type included), and the stage it is if it is a
/// shader. The console numbers amplification and mesh shaders 23 and 24
/// (PC: 24 and 25); its own types from 25 on are not sized yet.
fn subobject(kind: u32) -> Option<(usize, Option<&'static str>)> {
    let stage = |name| Some((24, Some(name)));
    match kind {
        1 => stage("vertex"),
        2 => stage("pixel"),
        3 => stage("domain"),
        4 => stage("hull"),
        5 => stage("geometry"),
        6 => stage("compute"),
        23 => stage("amplification"),
        24 => stage("mesh"),
        0 => Some((16, None)),
        12 | 19 => Some((24, None)),
        7 | 15 => Some((40, None)),
        8 => Some((336, None)),
        10 => Some((48, None)),
        11 => Some((56, None)),
        17 => Some((16, None)),
        21 => Some((64, None)),
        22 => Some((32, None)),
        9 | 13 | 14 | 16 | 18 | 20 => Some((8, None)),
        _ => None,
    }
}

/// Device method: a pipeline from a stream; the shaders are kept, and the
/// graphics state for the host pipeline.
///
/// # Safety
///
/// `desc` is null or valid for reads of a [`StreamDesc`] whose `stream`
/// points to `size` bytes laid out as subobjects, each whole at its
/// position and, for a shader, with each shader's `code` is null or valid
/// for reads of its `size` bytes. `iid` is null or points to an interface
/// id valid for reads, and `out` is null or valid for a write of an object
/// pointer, for the duration of the call.
pub(crate) unsafe extern "system" fn create_stream(
    _: This,
    desc: *const StreamDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: `desc` is null or readable, per this function's contract.
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if desc.stream.is_null() {
        return E_INVALIDARG;
    }
    let mut shaders = Vec::new();
    #[cfg(windows)]
    let mut graphics = host::GraphicsState::default();
    let mut at = 0;
    while at + 8 <= desc.size {
        // SAFETY: inside the stream, per this function's contract.
        let kind = unsafe { desc.stream.add(at).cast::<u32>().read_unaligned() };
        let Some((size, stage)) = subobject(kind) else {
            tracing::warn!(kind, at, "pipeline subobject of unknown size; the rest is ignored");
            break;
        };
        if let Some(stage) = stage {
            // SAFETY: a shader subobject has its bytecode at +8, and its
            // code is as this function's contract gives it.
            let code = unsafe { &*desc.stream.add(at + 8).cast::<Bytecode>() };
            // SAFETY: as above.
            shaders.extend(unsafe { copy(stage, code) });
        }
        #[cfg(windows)]
        // SAFETY: a whole subobject of this kind lies at `at`.
        unsafe {
            graphics.read(kind, desc.stream.add(at))
        };
        at += size;
    }
    // SAFETY: `iid` and `out` as this function's contract gives them.
    unsafe {
        hand_out(
            shaders,
            #[cfg(windows)]
            Some(Box::new(graphics)),
            iid,
            out,
        )
    }
}

/// `QueryInterface`.
///
/// # Safety
///
/// `this` is a live pipeline this module created. `iid` is null or points
/// to an interface id valid for reads, and `out` is null or valid for a
/// write of an object pointer, for the duration of the call.
unsafe extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &PIPELINE_IIDS).then_some(this);
    object::answer(PIPELINE, iid, found, out)
}
/// `Release`.
///
/// # Safety
///
/// `this` is a live pipeline this module created, holding the reference the
/// caller gives up; it must not be used after the last one is released.
unsafe extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Pipeline>(this)
}

/// IUnknown 3, object 4, device child 1, pipeline 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<PIPELINE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table(f)
};

#[cfg(windows)]
pub(crate) mod host;
#[cfg(windows)]
mod library;
#[cfg(windows)]
mod make;

/// Keeps compiled host pipelines, and the list of title pipelines to start
/// early, in `dir` (a folder of the cache) across runs.
#[cfg(windows)]
pub fn use_pipeline_cache(dir: std::path::PathBuf) {
    library::use_pipeline_cache(dir);
}

/// Saves the pipeline cache now if anything was added (at the end of a run).
#[cfg(windows)]
pub fn save_pipeline_cache() {
    library::save(true);
}

#[cfg(test)]
mod tests {
    #[cfg(all(windows, feature = "diagnostics"))]
    #[test]
    fn skip_hashes_accepts_full_hashes_only() {
        assert!(super::parse_skip_hashes(" , ").is_empty());
        assert_eq!(
            super::parse_skip_hashes(" 0123456789ABCDEF, 0123456789abcde, xyz, 0011223344556677,"),
            ["0123456789abcdef", "0011223344556677"]
        );
    }

    use super::*;
    use crate::object::{S_OK, slot};

    fn shaders(p: This) -> Vec<(&'static str, Vec<u8>)> {
        // SAFETY: `p` is a pipeline the test just created.
        unsafe { pipeline(p) }.shaders.iter().map(|s| (s.stage, s.container.clone())).collect()
    }

    #[test]
    fn pipelines_keep_their_shaders() {
        let (vs, ps) = ([1u8, 2, 3], [4u8; 5]);
        let none = || Bytecode { code: std::ptr::null(), size: 0 };
        let desc = GraphicsHead {
            root: std::ptr::null(),
            stages: [
                Bytecode { code: vs.as_ptr(), size: vs.len() },
                Bytecode { code: ps.as_ptr(), size: ps.len() },
                none(),
                none(),
                none(),
            ],
        };
        let mut p = std::ptr::null();
        // SAFETY: a local description whose code outlives the call; a static
        // id and a local to write.
        let made = unsafe { create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p) };
        assert_eq!(made, S_OK);
        let p = crate::test_support::valid(p);
        assert_eq!(shaders(p), [("vertex", vs.to_vec()), ("pixel", ps.to_vec())]);
        // A second pipeline with the same pixel shader shares its container.
        let mut q = std::ptr::null();
        // SAFETY: as above.
        unsafe { create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut q) };
        let q = crate::test_support::valid(q);
        // SAFETY: `p` and `q` are pipelines the test just created.
        let (sp, sq) = unsafe { (pipeline(p), pipeline(q)) };
        assert!(Arc::ptr_eq(&sp.shaders[1], &sq.shaders[1]));
        let release = slot::<unsafe extern "system" fn(This) -> u32>(p, 0x10);
        // SAFETY: `p` is live; this releases its only reference.
        assert_eq!(unsafe { release(p) }, 0);
    }

    /// The stream layout Grand Theft Auto VI builds (title+0x86f090): root
    /// signature, five shader slots (the first a mesh shader), blend, sample
    /// mask, rasterizer, depth-stencil 1, sample description, input layout,
    /// topology, render target and depth formats: 0x2b0 bytes.
    #[test]
    fn streams_yield_their_shaders() {
        let (ms, ps) = ([9u8; 7], [4u8; 5]);
        let mut stream = vec![0u8; 0x2b0];
        let mut put = |at: usize, v: u64| stream[at..at + 8].copy_from_slice(&v.to_le_bytes());
        for (at, kind) in [
            (0x10, 24),
            (0x28, 2),
            (0x40, 3),
            (0x58, 4),
            (0x70, 5),
            (0x88, 8),
            (0x1d8, 9),
            (0x1e0, 10),
            (0x210, 21),
            (0x250, 17),
            (0x260, 12),
            (0x278, 14),
            (0x280, 15),
            (0x2a8, 16),
        ] {
            put(at, kind);
        }
        put(0x18, ms.as_ptr() as u64);
        put(0x20, ms.len() as u64);
        put(0x30, ps.as_ptr() as u64);
        put(0x38, ps.len() as u64);
        // Topology type 3 (triangles), one render target of format 28.
        put(0x278, 14 | 3 << 32);
        put(0x280, 15 | 28 << 32);
        put(0x2a0, 1 << 32);
        let desc = StreamDesc { size: stream.len(), stream: stream.as_ptr() };
        let mut p = std::ptr::null();
        // SAFETY: `stream` is a local of the size the description says, and
        // the code it points at outlives the call.
        let hr = unsafe { create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p) };
        assert_eq!(hr, S_OK);
        let p = crate::test_support::valid(p);
        assert_eq!(shaders(p), [("mesh", ms.to_vec()), ("pixel", ps.to_vec())]);
        #[cfg(windows)]
        {
            // SAFETY: `p` is a pipeline the test just created.
            let g = unsafe { pipeline(p) }.host.graphics.as_deref().unwrap();
            assert_eq!(g.topology.0, 3);
            assert_eq!((g.targets.NumRenderTargets, g.targets.RTFormats[0].0), (1, 28));
        }
    }
}

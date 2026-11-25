//! Translated shaders kept on disk, so a run does not translate again what
//! an earlier one did: under the configured
//! cache folder, keyed by the container's content and by the VIctory build
//! that translated it (the BLAKE3 of our executable), so a rebuilt
//! translator never serves an old translation. Entries go through
//! `victory-pipeline`'s content-addressed cache, which writes them
//! atomically; an entry that does not read back is translated again, and a
//! cache that cannot be read or written is translated around (logged), so
//! only the translator decides that a shader has no translation.
//!
//! The entry is the bytes of [`HostShader::to_bytes`], whose format
//! `victory-shader` owns.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use victory_pipeline::{Cache, Converter};
use victory_shader::ShaderError;
use victory_shader::emit::HostShader;

static CACHE: OnceLock<Cache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// Keeps translated shaders in `dir` from now on.
pub fn use_shader_cache(dir: PathBuf) {
    match Cache::open(dir) {
        Ok(cache) => {
            let _ = CACHE.set(cache);
        }
        Err(e) => tracing::warn!(error = %e, "no shader cache"),
    }
}

/// Shaders found in the cache and translated since the start.
pub(crate) fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// The most lanes a host wave has, which translations depend on: the host
/// device's, or 64 (nothing split) without one.
fn host_lanes() -> u32 {
    crate::host::host().map_or(victory_shader::emit::CONSOLE_WAVE, |h| h.wave_lanes)
}

/// The BLAKE3 of our executable file and of the translation settings it
/// runs with: the loop limit, which a diagnostics build can change at run
/// time, and the host's wave lanes, which decide how wave64 programs are
/// translated.
fn fingerprint(exe: &[u8], limit: u32, lanes: u32) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(exe);
    h.update(&limit.to_le_bytes());
    h.update(&lanes.to_le_bytes());
    *h.finalize().as_bytes()
}

/// What identifies this build and host ([`fingerprint`]).
fn build() -> &'static [u8; 32] {
    static BUILD: OnceLock<[u8; 32]> = OnceLock::new();
    BUILD.get_or_init(|| {
        let exe = std::env::current_exe().and_then(std::fs::read);
        let limit = victory_shader::emit::loop_limit();
        if limit != victory_shader::emit::DEFAULT_LOOP_LIMIT {
            tracing::warn!(limit, "VICTORY_SHADER_LOOP_LIMIT: translated loops stop sooner");
        }
        match exe {
            Ok(bytes) => fingerprint(&bytes, limit, host_lanes()),
            Err(e) => {
                // Without it no entry can be trusted: a fresh key per run.
                tracing::warn!(error = %e, "cannot read our executable; shader cache unused");
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default();
                *blake3::hash(&now.as_nanos().to_le_bytes()).as_bytes()
            }
        }
    })
}

/// The translation of a console shader container, from the cache when
/// there is one. Only the translator refuses a shader: a cache that fails
/// is translated around.
pub(crate) fn host_shader(container: &[u8]) -> Result<HostShader, ShaderError> {
    let lanes = host_lanes();
    let translate = |c: &[u8]| victory_shader::emit::translate(c, lanes);
    match CACHE.get() {
        Some(cache) => cached(cache, build(), container, translate),
        None => translate(container),
    }
}

/// Logs a cache failure: the first at warn, the rest at debug (a broken
/// cache fails for every shader).
fn cache_failed(e: &anyhow::Error) {
    static FAILED: crate::warn::Warning = crate::warn::Warning::new(
        concat!(file!(), ":", line!()),
        "shader cache failed; translating without it",
    );
    let mut first = false;
    FAILED.hit(|| {
        first = true;
        tracing::warn!(error = format!("{e:#}"), "shader cache failed; translating without it");
    });
    if !first {
        tracing::debug!(error = format!("{e:#}"), "shader cache failed; translating without it");
    }
}

/// The cache entry's converter: the build and the container in, the
/// encoded translation out. `ran` tells a miss from a hit.
struct Translate<F> {
    translate: F,
    ran: AtomicBool,
}

impl<F: Fn(&[u8]) -> Result<HostShader, ShaderError> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        "shader.dxil"
    }
    fn version(&self) -> u32 {
        // The build fingerprint in the input does the versioning.
        1
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        // Typed, so `cached` can tell the translator's refusal from the
        // cache's own failure.
        let shader = (self.translate)(&input[32..]).map_err(anyhow::Error::new)?;
        Ok(shader.to_bytes())
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, ShaderError> + Send + Sync,
) -> Result<HostShader, ShaderError> {
    let input = [&build[..], container].concat();
    let converter = Translate { translate, ran: AtomicBool::new(false) };
    for _ in 0..2 {
        let path = match cache.get_or_convert(&converter, &input) {
            Ok(path) => path,
            Err(e) => match e.downcast::<ShaderError>() {
                Ok(refused) => return Err(refused),
                Err(e) => {
                    cache_failed(&e);
                    break;
                }
            },
        };
        if let Some(shader) = std::fs::read(&path).ok().and_then(|b| HostShader::from_bytes(&b)) {
            let (counter, what) = if converter.ran.load(Ordering::Relaxed) {
                (&MISSES, "translated and cached")
            } else {
                (&HITS, "from the cache")
            };
            counter.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader {what}");
            return Ok(shader);
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        if let Err(e) = std::fs::remove_file(&path) {
            tracing::warn!(path = %path.display(), error = %e, "cannot remove an unreadable shader cache entry");
        }
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    (converter.translate)(container)
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_shader::emit::{StaticSampler, UserData};
    use victory_shader::{HwStage, RootType};

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData { index: 8, slot: 1, root: RootType::Table, dword: 0 }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([0x400, 0xc0_0000, 0x18f0_0000, 0])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            used: Vec::new(),
            wave_size: Some(64),
        }
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build translates again.
        assert_eq!(cached(&cache, &[8; 32], container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH1 broken").unwrap();
        }
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        // Failures are not stored.
        let fail = |_: &[u8]| Err::<HostShader, _>(ShaderError::NotDxbc);
        assert!(cached(&cache, &build, b"other", fail).is_err());
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    /// A translation the translator refuses comes back through the cache
    /// with the translator's own error, not as text.
    #[test]
    fn a_refused_translation_keeps_its_error() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let unsupported = || ShaderError::Unsupported { offset: 0x40, what: "v_example" };
        let refuse = |_: &[u8]| Err::<HostShader, _>(unsupported());
        let refused = cached(&cache, &[7; 32], b"a container", refuse);
        assert_eq!(refused, Err(unsupported()));
    }

    /// A cache that cannot store the translation still hands it over: the
    /// shader translated, so its draws must not be dropped.
    #[test]
    fn a_cache_that_cannot_store_still_returns_the_translation() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("cache");
        let cache = Cache::open(&root).unwrap();
        // A file where the cache's folder was: nothing can be written there.
        std::fs::remove_dir(&root).unwrap();
        std::fs::write(&root, b"not a folder").unwrap();
        let translate = |_: &[u8]| Ok(shader());
        assert_eq!(cached(&cache, &[7; 32], b"a container", translate), Ok(shader()));
    }

    /// A host of other wave lanes translates differently: its entries are
    /// apart, as another build's are.
    #[test]
    fn the_host_wave_lanes_key_the_cache() {
        let exe = b"our executable";
        assert_ne!(fingerprint(exe, 1 << 20, 32), fingerprint(exe, 1 << 20, 64));
        assert_eq!(fingerprint(exe, 1 << 20, 32), fingerprint(exe, 1 << 20, 32));
        assert_ne!(fingerprint(exe, 1 << 20, 32), fingerprint(exe, 1 << 19, 32));
    }
}

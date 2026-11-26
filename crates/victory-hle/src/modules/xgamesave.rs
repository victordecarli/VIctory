//! XGameSave: the user's game saves, kept on the host under the title's
//! data directory, next to persistent local storage (`runtime::Title::
//! saves`), never in the cache. A provider is a folder per user and
//! service configuration, a container a folder in it.
//!
//! An update is all or nothing, as documented: each one writes a complete
//! new version of the container (`v<N>.new/`, unchanged blobs hard-linked
//! from the previous version), flushed to disk, and one directory rename
//! commits it (`v<N>`); the highest committed version is the container.
//! A crash leaves the old version or the new one, never a mix; leftovers
//! are removed on sight. Deleting a container renames it away first.
//!
//! Names are exact: a version's `manifest` holds the container's name, its
//! display name and the blob names (blob `i` is the file `i`), so case,
//! characters the host forbids, reserved device names and length never
//! reach a file name. A container's folder is its name made safe (and
//! shortened) plus a hash of the exact name, so names differing only in
//! case get different folders. Offline there is no cloud copy, so nothing
//! ever needs a sync.

use std::ffi::{CString, c_char, c_void};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use super::xasync::{self, AsyncBlock, E_INVALIDARG};
use super::xgameruntime::{
    Guid, HRESULT, Interface, Methods, Object, S_OK, add_ref, missing, put, text,
};
use super::{storage, xuser};
use crate::runtime;

pub const PROVIDER: Guid =
    Guid::new(0x704c_3f58, 0xe629, 0x4cc2, [0xb1, 0x97, 0x30, 0x51, 0x1b, 0x99, 0x6f, 0xe2]);
/// As Grand Theft Auto VI (2026) asks for it; every slot it calls takes
/// the public API's arguments in the same order as under the provider's
/// own id (older titles).
pub const IID_2026: Guid =
    Guid::new(0x1bff_f3af, 0xf14a, 0x40a3, [0x8e, 0x35, 0x9a, 0xda, 0x90, 0x65, 0x93, 0xf9]);

pub static OBJECT: Object = Object::new("game save", &VTABLE);
pub const INTERFACES: &[Interface] =
    &[(PROVIDER, PROVIDER, &OBJECT), (PROVIDER, IID_2026, &OBJECT)];

// The public API's error codes.
pub const INVALID_CONTAINER_NAME: HRESULT = 0x8083_0001_u32 as i32;
pub const NO_ACCESS: HRESULT = 0x8083_0002_u32 as i32;
pub const OUT_OF_LOCAL_STORAGE: HRESULT = 0x8083_0003_u32 as i32;
pub const UPDATE_TOO_BIG: HRESULT = 0x8083_0005_u32 as i32;
pub const QUOTA_EXCEEDED: HRESULT = 0x8083_0006_u32 as i32;
pub const BUFFER_TOO_SMALL: HRESULT = 0x8083_0007_u32 as i32;
pub const BLOB_NOT_FOUND: HRESULT = 0x8083_0008_u32 as i32;

/// Per user and title, as documented.
const QUOTA: u64 = 256 << 20;
/// What one update may write in all, as documented.
const MAX_UPDATE: usize = 16 << 20;
const MANIFEST: &str = "manifest";
/// Readable part of a container's folder name, at most.
const READABLE: usize = 100;

/// Every save operation, so none sees another's half-made version.
static SAVES: Mutex<()> = Mutex::new(());

fn saves_lock() -> std::sync::MutexGuard<'static, ()> {
    SAVES.lock().unwrap_or_else(|e| e.into_inner())
}

/// `name` as a file name: letters, digits, `-` and `_` stay, every other
/// byte becomes `%XX`. The result never contains a dot.
fn file_name(name: &str) -> String {
    name.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

/// The name behind a file name from [`file_name`]; `None` for other files.
fn decode(file: &str) -> Option<String> {
    if file.contains('.') {
        return None;
    }
    let mut out = Vec::new();
    let mut bytes = file.bytes();
    while let Some(b) = bytes.next() {
        if b == b'%' {
            let hex = [bytes.next()?, bytes.next()?];
            out.push(u8::from_str_radix(std::str::from_utf8(&hex).ok()?, 16).ok()?);
        } else {
            out.push(b);
        }
    }
    String::from_utf8(out).ok()
}

/// The container's folder: its name made safe, cut short, then `~` and a
/// hash of the exact name (64-bit FNV-1a, stable across builds).
fn container_dir(root: &Path, name: &str) -> PathBuf {
    let hash = name
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325_u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3));
    let mut readable = file_name(name);
    readable.truncate(READABLE); // ASCII only
    root.join(format!("{readable}~{hash:016x}"))
}

/// Documented: the path part (up to the last `/`) letters, digits, `_` and
/// `/`; the file part letters, digits, `_`, `.` and `-`, not empty, not
/// ending in `.`, no `..`; 256 characters at most.
fn valid_container(name: &str) -> bool {
    let (path, file) = name.rsplit_once('/').unwrap_or(("", name));
    name.len() <= 256
        && path.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'/')
        && !file.is_empty()
        && !file.ends_with('.')
        && !file.contains("..")
        && file.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'))
}

/// A committed version of a container.
struct Version {
    dir: PathBuf,
    number: u64,
    name: String,
    display: String,
    /// Blob names; blob `i` is the file `i`.
    blobs: Vec<String>,
}

impl Version {
    fn blob(&self, i: usize) -> PathBuf {
        self.dir.join(i.to_string())
    }

    fn sizes(&self) -> Vec<(String, u64)> {
        let size = |i| std::fs::metadata(self.blob(i)).map_or(0, |m| m.len());
        let blobs = self.blobs.iter().enumerate();
        blobs.map(|(i, n)| (n.clone(), size(i))).collect()
    }
}

/// The manifest: length-prefixed strings (u32 LE): container name, display
/// name, then each blob name.
fn manifest(name: &str, display: &str, blobs: &[&str]) -> Vec<u8> {
    let mut out = Vec::new();
    for s in [name, display].into_iter().chain(blobs.iter().copied()) {
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }
    out
}

fn parse_manifest(bytes: &[u8]) -> Option<Vec<String>> {
    let mut out = Vec::new();
    let mut rest = bytes;
    while !rest.is_empty() {
        let (len, tail) = rest.split_first_chunk::<4>()?;
        let len = u32::from_le_bytes(*len) as usize;
        out.push(String::from_utf8(tail.get(..len)?.to_vec()).ok()?);
        rest = &tail[len..];
    }
    (out.len() >= 2).then_some(out)
}

/// The container's newest committed version, if any. Removes what a crash
/// left: unfinished versions and older ones.
fn current(dir: &Path) -> std::io::Result<Option<Version>> {
    let mut committed = Vec::new();
    for entry in std::fs::read_dir(dir)?.flatten() {
        let file = entry.file_name();
        let file = file.to_string_lossy();
        if file.ends_with(".new") {
            std::fs::remove_dir_all(entry.path())?;
        } else if let Some(n) = file.strip_prefix('v').and_then(|n| n.parse::<u64>().ok()) {
            committed.push(n);
        }
    }
    committed.sort_unstable();
    let Some(&number) = committed.last() else {
        return Ok(None);
    };
    for old in &committed[..committed.len() - 1] {
        std::fs::remove_dir_all(dir.join(format!("v{old}")))?;
    }
    let version = dir.join(format!("v{number}"));
    let bad = || std::io::Error::new(std::io::ErrorKind::InvalidData, "bad save manifest");
    let mut names = parse_manifest(&std::fs::read(version.join(MANIFEST))?).ok_or_else(bad)?;
    let blobs = names.split_off(2);
    let [name, display] = <[String; 2]>::try_from(names).map_err(|_| bad())?;
    Ok(Some(Version { dir: version, number, name, display, blobs }))
}

/// The version of container `name` in `dir`; none for a missing container
/// or another name that hashes alike.
fn current_named(dir: &Path, name: &str) -> std::io::Result<Option<Version>> {
    match current(dir) {
        Ok(v) => Ok(v.filter(|v| v.name == name)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// Where a crash test stops the process (test builds only).
fn step() {
    #[cfg(test)]
    tests::crash_point();
}

/// A blob of a new version: new data, or the previous version's file.
enum Source<'a> {
    Data(&'a [u8]),
    Kept(PathBuf),
}

/// Writes version `number` of the container in `dir` and commits it.
fn commit(
    dir: &Path,
    number: u64,
    name: &str,
    display: &str,
    blobs: &[(&str, Source<'_>)],
) -> std::io::Result<()> {
    let staged = dir.join(format!("v{number}.new"));
    if staged.exists() {
        std::fs::remove_dir_all(&staged)?;
    }
    std::fs::create_dir_all(&staged)?;
    let write = |path: PathBuf, data: &[u8]| -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        file.write_all(data)?;
        file.sync_all()
    };
    for (i, (_, source)) in blobs.iter().enumerate() {
        let path = staged.join(i.to_string());
        match source {
            Source::Data(data) => write(path, data)?,
            // Files are never changed in place, so sharing one is safe.
            Source::Kept(from) => {
                std::fs::hard_link(from, &path).or_else(|_| std::fs::copy(from, &path).map(drop))?
            }
        }
        step();
    }
    let names: Vec<&str> = blobs.iter().map(|(n, _)| *n).collect();
    write(staged.join(MANIFEST), &manifest(name, display, &names))?;
    step();
    std::fs::rename(&staged, dir.join(format!("v{number}")))?;
    step();
    Ok(())
}

/// The containers under a provider's root, by exact name; containers in
/// the old one-file-per-blob layout are converted first.
fn containers(root: &Path) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(root).into_iter().flatten().flatten() {
        let path = entry.path();
        let file = entry.file_name().to_string_lossy().into_owned();
        if !path.is_dir() {
            continue;
        }
        if file.ends_with(".gone") {
            let _ = std::fs::remove_dir_all(&path);
        } else if file.contains('~') {
            if let Ok(Some(v)) = current(&path) {
                out.push((v.name, path));
            }
        } else if let Some(name) = decode(&file)
            && let Some(dir) = migrate(root, &name, &path)
        {
            out.push((name, dir));
        }
    }
    out.sort();
    out
}

/// Moves a container of the old layout (blob files named by encoded name,
/// a `.display` file) into a committed version, then removes it.
fn migrate(root: &Path, name: &str, old: &Path) -> Option<PathBuf> {
    let dir = container_dir(root, name);
    let done = current_named(&dir, name).ok()?.is_some(); // a crash after the commit
    if !done {
        let display = std::fs::read_to_string(old.join(".display")).unwrap_or_default();
        let mut blobs: Vec<(String, Vec<u8>)> = std::fs::read_dir(old)
            .ok()?
            .flatten()
            .filter_map(|e| {
                let blob = decode(e.file_name().to_str()?)?;
                Some((blob, std::fs::read(e.path()).ok()?))
            })
            .collect();
        blobs.sort();
        let sources: Vec<_> = blobs.iter().map(|(n, d)| (n.as_str(), Source::Data(d))).collect();
        commit(&dir, 1, name, &display, &sources).ok()?;
    }
    std::fs::remove_dir_all(old).ok()?;
    tracing::info!(container = name, "game save moved to the versioned layout");
    Some(dir)
}

fn used_bytes(root: &Path) -> u64 {
    containers(root)
        .iter()
        .filter_map(|(name, dir)| current_named(dir, name).ok().flatten())
        .map(|v| v.sizes().iter().map(|(_, size)| size).sum::<u64>())
        .sum()
}

struct Provider {
    root: PathBuf,
}
struct Container {
    name: String,
    dir: PathBuf,
}
struct Update {
    name: String,
    dir: PathBuf,
    display: Option<String>,
    /// Blob writes (`Some`) and deletions (`None`), in order.
    changes: Vec<(String, Option<Vec<u8>>)>,
}

/// A handle we gave out.
///
/// # Safety
///
/// `handle` is null or a handle this module gave out and the title has not
/// closed, and nothing frees it while the reference lives.
unsafe fn get<'a, T>(handle: *const T) -> Option<&'a T> {
    // SAFETY: a handle is a box we created and the title has not closed, per
    // this function's contract.
    unsafe { handle.as_ref() }
}
fn give<T>(value: T) -> *mut T {
    Box::into_raw(Box::new(value))
}
/// # Safety
///
/// `handle` is null or a handle this module gave out, closed once and not
/// used afterwards.
unsafe fn take<T>(handle: *mut T) {
    if !handle.is_null() {
        // SAFETY: as in `get`; closing ends the handle.
        drop(unsafe { Box::from_raw(handle) });
    }
}

impl Provider {
    /// The provider for the only user and `config` (the service
    /// configuration id), under `saves`. An empty id names no
    /// configuration, like none: as a folder name it would be the user's
    /// folder itself, where the other configurations' folders are.
    fn open(saves: &Path, config: Option<&str>) -> Result<Provider, HRESULT> {
        let config = config.filter(|c| !c.is_empty()).unwrap_or("default");
        let root = saves.join(format!("{:016X}", xuser::XUID)).join(file_name(config));
        std::fs::create_dir_all(&root).map_err(|_| NO_ACCESS)?;
        let _saves = saves_lock();
        containers(&root); // converts old containers now
        Ok(Provider { root })
    }

    fn remaining(&self) -> u64 {
        let _saves = saves_lock();
        remaining(&self.root)
    }

    fn container(&self, name: &str) -> Container {
        Container { name: name.to_owned(), dir: container_dir(&self.root, name) }
    }
}

/// Quota left under a provider's root, or less if the host disk is fuller.
fn remaining(root: &Path) -> u64 {
    QUOTA.saturating_sub(used_bytes(root)).min(storage::disk_free(root))
}

/// Why an update writing `written` bytes, `replaced` of them over blobs it
/// replaces, does not fit: past the per-user quota (`used` bytes kept), or
/// past the host disk's `free` bytes, which is not the user's quota (the
/// new version is written before the old one goes, so nothing it replaces
/// counts there).
fn no_room(written: u64, replaced: u64, used: u64, free: u64) -> Option<HRESULT> {
    if written > QUOTA.saturating_sub(used) + replaced {
        Some(QUOTA_EXCEEDED)
    } else if written > free {
        Some(OUT_OF_LOCAL_STORAGE)
    } else {
        None
    }
}

fn io_error(dir: &Path, e: &std::io::Error) -> HRESULT {
    tracing::warn!(dir = %dir.display(), %e, "game save");
    if e.kind() == std::io::ErrorKind::StorageFull { OUT_OF_LOCAL_STORAGE } else { NO_ACCESS }
}

impl Update {
    /// Applies the update as one new version of the container.
    fn submit(&self) -> HRESULT {
        let _saves = saves_lock();
        let current = match current(&self.dir) {
            Ok(v) => v,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return io_error(&self.dir, &e),
        };
        if current.as_ref().is_some_and(|v| v.name != self.name) {
            // Another container's name hashes alike: never mix the two.
            return NO_ACCESS;
        }
        // The blobs after the update: name → new data or the kept file.
        let mut blobs: Vec<(&str, Source<'_>)> = current
            .iter()
            .flat_map(|v| {
                let kept = v.blobs.iter().enumerate();
                kept.map(|(i, n)| (n.as_str(), Source::Kept(v.blob(i))))
            })
            .collect();
        for (name, data) in &self.changes {
            blobs.retain(|(n, _)| n != name);
            if let Some(data) = data {
                blobs.push((name, Source::Data(data)));
            }
        }
        blobs.sort_by(|a, b| a.0.cmp(b.0));
        let written: usize = blobs
            .iter()
            .map(|(_, s)| match s {
                Source::Data(d) => d.len(),
                Source::Kept(_) => 0,
            })
            .sum();
        if written > MAX_UPDATE {
            return UPDATE_TOO_BIG;
        }
        let replaced: u64 = current.as_ref().map_or(0, |v| {
            let sizes = v.sizes();
            let changed = sizes.iter().filter(|(n, _)| self.changes.iter().any(|(c, _)| c == n));
            changed.map(|(_, size)| size).sum()
        });
        let root = self.dir.parent().unwrap_or(&self.dir);
        let free = storage::disk_free(root);
        if let Some(hr) = no_room(written as u64, replaced, used_bytes(root), free) {
            return hr;
        }
        let display = self.display.as_deref();
        let display = display.or(current.as_ref().map(|v| v.display.as_str()));
        let number = current.as_ref().map_or(1, |v| v.number + 1);
        if let Err(e) = commit(&self.dir, number, &self.name, display.unwrap_or(""), &blobs) {
            return io_error(&self.dir, &e);
        }
        // The old version is garbage now; a crash here leaves it for later.
        if let Some(old) = &current {
            let _ = std::fs::remove_dir_all(&old.dir);
            step();
        }
        S_OK
    }
}

/// Deletes a container: renamed away (one step), then removed.
fn delete(root: &Path, name: &str) -> HRESULT {
    let _saves = saves_lock();
    let dir = container_dir(root, name);
    match current_named(&dir, name) {
        Ok(Some(_)) => {}
        Ok(None) => return S_OK,
        Err(e) => return io_error(&dir, &e),
    }
    let gone = dir.with_extension("gone");
    let _ = std::fs::remove_dir_all(&gone);
    step();
    if let Err(e) = std::fs::rename(&dir, &gone) {
        return io_error(&dir, &e);
    }
    step();
    let _ = std::fs::remove_dir_all(&gone);
    S_OK
}

/// Reads the named blobs (all of them for `None`) of a container.
fn read(c: &Container, names: Option<Vec<String>>) -> Result<Vec<(String, Vec<u8>)>, HRESULT> {
    let _saves = saves_lock();
    let version =
        current_named(&c.dir, &c.name).map_err(|e| io_error(&c.dir, &e))?.ok_or(BLOB_NOT_FOUND)?;
    let names = names.unwrap_or_else(|| version.blobs.clone());
    names
        .into_iter()
        .map(|name| {
            let i = version.blobs.iter().position(|b| *b == name).ok_or(BLOB_NOT_FOUND)?;
            let data = std::fs::read(version.blob(i)).map_err(|e| io_error(&c.dir, &e))?;
            Ok((name, data))
        })
        .collect()
}

/// One blob as the public API lays it out: name, size, data.
const BLOB: usize = 24;

/// Blobs laid out as the title reads them: the blob array, then each name
/// (NUL-terminated) and its data, with pointers `base` plus offset.
fn layout(blobs: &[(String, Vec<u8>)], base: usize) -> Vec<u8> {
    let mut out = vec![0u8; blobs.len() * BLOB];
    for (i, (name, data)) in blobs.iter().enumerate() {
        let name_at = base + out.len();
        out.extend_from_slice(name.as_bytes());
        out.push(0);
        let data_at = base + out.len();
        out.extend_from_slice(data);
        let entry = &mut out[i * BLOB..(i + 1) * BLOB];
        entry[..8].copy_from_slice(&name_at.to_le_bytes());
        entry[8..12].copy_from_slice(&(data.len() as u32).to_le_bytes());
        entry[16..].copy_from_slice(&data_at.to_le_bytes());
    }
    out
}

/// The names a title passes (`count` strings, or all blobs if null).
///
/// # Safety
///
/// `names` is null or valid for reads of `count` pointers, each a
/// NUL-terminated string or null.
unsafe fn names(names: *const *const c_char, count: u32) -> Option<Vec<String>> {
    if names.is_null() {
        return None;
    }
    // SAFETY: non-null, and `count` strings, per this function's contract.
    let names = unsafe { std::slice::from_raw_parts(names, count as usize) };
    Some(names.iter().map(|&n| text(n).unwrap_or_default()).collect())
}

type This = *const Object;
type ContainerCallback = extern "system" fn(*const ContainerInfo, *mut c_void) -> bool;
type BlobCallback = extern "system" fn(*const BlobInfo, *mut c_void) -> bool;

#[repr(C)]
pub struct ContainerInfo {
    pub name: *const c_char,
    pub display_name: *const c_char,
    pub blob_count: u32,
    pub total_size: u64,
    pub last_modified: i64,
    pub needs_sync: bool,
}

#[repr(C)]
pub struct BlobInfo {
    pub name: *const c_char,
    pub size: u32,
}

/// Calls `callback` with the information of each container that passes
/// `keep` until it returns false; never under our lock (it may call us).
///
/// # Safety
///
/// `context` is whatever `callback`, a function of the title's, accepts as
/// its second argument, and `callback` is valid to call with it.
unsafe fn report(
    root: &Path,
    keep: impl Fn(&str) -> bool,
    context: *mut c_void,
    callback: ContainerCallback,
) {
    let infos: Vec<(CString, CString, u32, u64, i64)> = {
        let _saves = saves_lock();
        containers(root)
            .into_iter()
            .filter(|(name, _)| keep(name))
            .filter_map(|(name, dir)| current_named(&dir, &name).ok().flatten())
            .map(|v| {
                let modified = std::fs::metadata(v.dir.join(MANIFEST))
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_secs() as i64);
                let size = v.sizes().iter().map(|(_, s)| s).sum();
                let count = v.blobs.len() as u32;
                (cstring(&v.name), cstring(&v.display), count, size, modified)
            })
            .collect()
    };
    for (name, display, blob_count, total_size, last_modified) in &infos {
        let info = ContainerInfo {
            name: name.as_ptr(),
            display_name: display.as_ptr(),
            blob_count: *blob_count,
            total_size: *total_size,
            last_modified: *last_modified,
            needs_sync: false,
        };
        if !callback(&info, context) {
            break;
        }
    }
}

fn cstring(s: &str) -> CString {
    CString::new(s.replace('\0', "")).unwrap_or_default()
}

static INIT: u8 = 0;
static QUOTA_CALL: u8 = 0;
static DELETE: u8 = 0;
static READ: u8 = 0;
static SUBMIT: u8 = 0;

/// # Safety
///
/// `_user`, `config`, `block` are null or valid as the API this stands in
/// for requires: a structure or block of its type, a NUL-terminated string,
/// or a buffer for its stated size, for the duration of the call.
unsafe extern "system" fn initialize_async(
    _: This,
    _user: *const c_void,
    config: *const c_char,
    _sync_on_demand: bool,
    block: *mut AsyncBlock,
) -> HRESULT {
    let saves = runtime::title().saves();
    let result = if saves.parent().is_none_or(|p| p.as_os_str().is_empty()) {
        Err(NO_ACCESS)
    } else {
        Provider::open(&saves, text(config).as_deref()).map(|p| xasync::bytes(give(p)))
    };
    // SAFETY: the pointers passed on are under this function's own contract.
    unsafe { xasync::finished_with(block, (&raw const INIT).cast(), result) }
}
/// # Safety
///
/// `block`, `out` are null or valid as the API this stands in for requires:
/// a structure or block of its type, a NUL-terminated string, or a buffer
/// for its stated size, for the duration of the call.
unsafe extern "system" fn initialize_result(
    _: This,
    block: *mut AsyncBlock,
    out: *mut *mut Provider,
) -> HRESULT {
    // SAFETY: the pointers passed on are under this function's own contract.
    unsafe { xasync::result(block, (&raw const INIT).cast(), out) }
}
/// # Safety
///
/// `provider` is null or valid as the API this stands in for requires: a
/// structure or block of its type, a NUL-terminated string, or a buffer for
/// its stated size, for the duration of the call.
unsafe extern "system" fn close_provider(_: This, provider: *mut Provider) {
    // SAFETY: as this function's contract gives them.
    unsafe { take(provider) };
}

/// # Safety
///
/// `provider`, `block` are null or valid as the API this stands in for
/// requires: a structure or block of its type, a NUL-terminated string, or
/// a buffer for its stated size, for the duration of the call.
unsafe extern "system" fn quota_async(
    _: This,
    provider: *const Provider,
    block: *mut AsyncBlock,
) -> HRESULT {
    // SAFETY: as this function's contract gives them.
    let result =
        unsafe { get(provider) }.ok_or(E_INVALIDARG).map(|p| xasync::bytes(p.remaining() as i64));
    // SAFETY: the pointers passed on are under this function's own contract.
    unsafe { xasync::finished_with(block, (&raw const QUOTA_CALL).cast(), result) }
}
/// # Safety
///
/// `block`, `out` are null or valid as the API this stands in for requires:
/// a structure or block of its type, a NUL-terminated string, or a buffer
/// for its stated size, for the duration of the call.
unsafe extern "system" fn quota_result(_: This, block: *mut AsyncBlock, out: *mut i64) -> HRESULT {
    // SAFETY: the pointers passed on are under this function's own contract.
    unsafe { xasync::result(block, (&raw const QUOTA_CALL).cast(), out) }
}

/// # Safety
///
/// `provider`, `name`, `block` are null or valid as the API this stands in
/// for requires: a structure or block of its type, a NUL-terminated string,
/// or a buffer for its stated size, for the duration of the call.
unsafe extern "system" fn delete_async(
    _: This,
    provider: *const Provider,
    name: *const c_char,
    block: *mut AsyncBlock,
) -> HRESULT {
    // SAFETY: as this function's contract gives them.
    let hr = match (unsafe { get(provider) }, text(name)) {
        (Some(p), Some(name)) if valid_container(&name) => delete(&p.root, &name),
        (Some(_), Some(_)) => INVALID_CONTAINER_NAME,
        _ => E_INVALIDARG,
    };
    // SAFETY: the pointers passed on are under this function's own contract.
    unsafe { xasync::finished(block, (&raw const DELETE).cast(), hr) }
}
/// # Safety
///
/// `block` is null or valid as the API this stands in for requires: a
/// structure or block of its type, a NUL-terminated string, or a buffer for
/// its stated size, for the duration of the call.
unsafe extern "system" fn delete_result(_: This, block: *mut AsyncBlock) -> HRESULT {
    let none = std::ptr::null_mut();
    // SAFETY: the pointers passed on are under this function's own contract.
    unsafe { xasync::XAsyncGetResult(block, (&raw const DELETE).cast(), 0, none, none.cast()) }
}

/// Reports the container if it exists; a missing one is no error, the
/// callback just does not run.
///
/// # Safety
///
/// `provider`, `name`, `context` are null or valid as the API this stands
/// in for requires: a structure or block of its type, a NUL-terminated
/// string, or a buffer for its stated size, for the duration of the call.
unsafe extern "system" fn container_info(
    _: This,
    provider: *const Provider,
    name: *const c_char,
    context: *mut c_void,
    callback: Option<ContainerCallback>,
) -> HRESULT {
    // SAFETY: as this function's contract gives them.
    let (Some(p), Some(name), Some(callback)) = (unsafe { get(provider) }, text(name), callback)
    else {
        return E_INVALIDARG;
    };
    if !valid_container(&name) {
        return INVALID_CONTAINER_NAME;
    }
    // SAFETY: as this function's contract gives them.
    unsafe { report(&p.root, |n| n == name, context, callback) };
    S_OK
}

/// # Safety
///
/// `provider`, `context` are null or valid as the API this stands in for
/// requires: a structure or block of its type, a NUL-terminated string, or
/// a buffer for its stated size, for the duration of the call.
unsafe extern "system" fn enumerate_containers(
    _: This,
    provider: *const Provider,
    context: *mut c_void,
    callback: Option<ContainerCallback>,
) -> HRESULT {
    // SAFETY: as this function's contract gives them.
    let (Some(p), Some(callback)) = (unsafe { get(provider) }, callback) else {
        return E_INVALIDARG;
    };
    // SAFETY: as this function's contract gives them.
    unsafe { report(&p.root, |_| true, context, callback) };
    S_OK
}

/// The container exists once an update to it is submitted.
///
/// # Safety
///
/// `provider`, `name`, `out` are null or valid as the API this stands in
/// for requires: a structure or block of its type, a NUL-terminated string,
/// or a buffer for its stated size, for the duration of the call.
unsafe extern "system" fn create_container(
    _: This,
    provider: *const Provider,
    name: *const c_char,
    out: *mut *mut Container,
) -> HRESULT {
    // SAFETY: as this function's contract gives them.
    let (Some(p), Some(name)) = (unsafe { get(provider) }, text(name)) else {
        return E_INVALIDARG;
    };
    if !valid_container(&name) {
        return INVALID_CONTAINER_NAME;
    }
    put(out, give(p.container(&name)))
}
/// # Safety
///
/// `container` is null or valid as the API this stands in for requires: a
/// structure or block of its type, a NUL-terminated string, or a buffer for
/// its stated size, for the duration of the call.
unsafe extern "system" fn close_container(_: This, container: *mut Container) {
    // SAFETY: as this function's contract gives them.
    unsafe { take(container) };
}

/// # Safety
///
/// `container`, `prefix`, `context` are null or valid as the API this
/// stands in for requires: a structure or block of its type, a
/// NUL-terminated string, or a buffer for its stated size, for the duration
/// of the call.
unsafe extern "system" fn enumerate_blobs(
    _: This,
    container: *const Container,
    prefix: *const c_char,
    context: *mut c_void,
    callback: Option<BlobCallback>,
) -> HRESULT {
    // SAFETY: as this function's contract gives them.
    let (Some(c), Some(callback)) = (unsafe { get(container) }, callback) else {
        return E_INVALIDARG;
    };
    let prefix = text(prefix).unwrap_or_default();
    let blobs = {
        let _saves = saves_lock();
        let version = current_named(&c.dir, &c.name).ok().flatten();
        version.map(|v| v.sizes()).unwrap_or_default()
    };
    for (name, size) in blobs.into_iter().filter(|(n, _)| n.starts_with(&prefix)) {
        let name = cstring(&name);
        let info = BlobInfo { name: name.as_ptr(), size: size as u32 };
        if !callback(&info, context) {
            break;
        }
    }
    S_OK
}

/// # Safety
///
/// `container`, `blob_names`, `count`, `out` are null or valid as the API
/// this stands in for requires: a structure or block of its type, a
/// NUL-terminated string, or a buffer for its stated size, for the duration
/// of the call.
unsafe extern "system" fn read_blobs(
    _: This,
    container: *const Container,
    blob_names: *const *const c_char,
    count: *mut u32,
    size: usize,
    out: *mut u8,
) -> HRESULT {
    // SAFETY: as this function's contract gives them.
    let found = unsafe { get(container) };
    let Some(c) = found else {
        return E_INVALIDARG;
    };
    // SAFETY: null or readable, per this function's contract.
    let wanted = unsafe { count.as_ref() }.copied().unwrap_or(0);
    // SAFETY: as this function's contract gives them.
    let blobs = match read(c, unsafe { names(blob_names, wanted) }) {
        Ok(blobs) => blobs,
        Err(hr) => return hr,
    };
    let bytes = layout(&blobs, out as usize);
    if out.is_null() || size < bytes.len() {
        return BUFFER_TOO_SMALL;
    }
    // SAFETY: non-null, and `size` writable bytes, per this function's
    // contract.
    unsafe { out.copy_from_nonoverlapping(bytes.as_ptr(), bytes.len()) };
    put(count, blobs.len() as u32)
}

/// Read at once; the result holds the layout with pointers relative to the
/// title's buffer, fixed up when the title takes it.
///
/// # Safety
///
/// `container`, `blob_names`, `block` are null or valid as the API this
/// stands in for requires: a structure or block of its type, a
/// NUL-terminated string, or a buffer for its stated size, for the duration
/// of the call.
unsafe extern "system" fn read_async(
    _: This,
    container: *const Container,
    blob_names: *const *const c_char,
    count: u32,
    block: *mut AsyncBlock,
) -> HRESULT {
    // SAFETY: as this function's contract gives them.
    let result = match unsafe { get(container) } {
        // SAFETY: as this function's contract gives them.
        Some(c) => read(c, unsafe { names(blob_names, count) }).map(|blobs| layout(&blobs, 0)),
        None => Err(E_INVALIDARG),
    };
    // SAFETY: the pointers passed on are under this function's own contract.
    unsafe { xasync::finished_with(block, (&raw const READ).cast(), result) }
}
/// # Safety
///
/// `block`, `out`, `count` are null or valid as the API this stands in for
/// requires: a structure or block of its type, a NUL-terminated string, or
/// a buffer for its stated size, for the duration of the call.
unsafe extern "system" fn read_result(
    _: This,
    block: *mut AsyncBlock,
    size: usize,
    out: *mut u8,
    count: *mut u32,
) -> HRESULT {
    let mut used = 0;
    // SAFETY: the pointers passed on are under this function's own contract.
    let hr = unsafe {
        xasync::XAsyncGetResult(block, (&raw const READ).cast(), size, out.cast(), &mut used)
    };
    // The documented error for a small buffer; the result stays for a
    // retry with a larger one.
    if hr == xasync::E_NOT_SUFFICIENT_BUFFER {
        return BUFFER_TOO_SMALL;
    }
    if hr < 0 {
        return hr;
    }
    // The first name follows the blob array, so its offset gives the count.
    let blobs = if used == 0 {
        0
    } else {
        // SAFETY: the result holds at least one blob entry.
        unsafe { out.cast::<usize>().read_unaligned() / BLOB }
    };
    for i in 0..blobs {
        for field in [0, 16] {
            // SAFETY: within the blob array just written.
            unsafe {
                let p = out.add(i * BLOB + field).cast::<usize>();
                p.write_unaligned(p.read_unaligned() + out as usize);
            }
        }
    }
    put(count, blobs as u32)
}

/// # Safety
///
/// `container`, `display`, `out` are null or valid as the API this stands
/// in for requires: a structure or block of its type, a NUL-terminated
/// string, or a buffer for its stated size, for the duration of the call.
unsafe extern "system" fn create_update(
    _: This,
    container: *const Container,
    display: *const c_char,
    out: *mut *mut Update,
) -> HRESULT {
    // SAFETY: as this function's contract gives them.
    let found = unsafe { get(container) };
    let Some(c) = found else {
        return E_INVALIDARG;
    };
    put(
        out,
        give(Update {
            name: c.name.clone(),
            dir: c.dir.clone(),
            display: text(display),
            changes: Vec::new(),
        }),
    )
}
/// # Safety
///
/// `update` is null or valid as the API this stands in for requires: a
/// structure or block of its type, a NUL-terminated string, or a buffer for
/// its stated size, for the duration of the call.
unsafe extern "system" fn close_update(_: This, update: *mut Update) {
    // SAFETY: as this function's contract gives them.
    unsafe { take(update) };
}

/// # Safety
///
/// `update`, `name`, `data` are null or valid as the API this stands in for
/// requires: a structure or block of its type, a NUL-terminated string, or
/// a buffer for its stated size, for the duration of the call.
unsafe extern "system" fn write_blob(
    _: This,
    update: *mut Update,
    name: *const c_char,
    data: *const u8,
    size: usize,
) -> HRESULT {
    // SAFETY: a handle we gave out, per this function's contract; the title
    // does not use it concurrently.
    let (Some(update), Some(name)) = (unsafe { update.as_mut() }, text(name)) else {
        return E_INVALIDARG;
    };
    if name.is_empty() || (data.is_null() && size > 0) {
        return E_INVALIDARG;
    }
    if size > MAX_UPDATE {
        return UPDATE_TOO_BIG;
    }
    let data = if size == 0 {
        Vec::new()
    } else {
        // SAFETY: non-null, and `size` readable bytes, per this function's
        // contract.
        unsafe { std::slice::from_raw_parts(data, size) }.to_vec()
    };
    update.changes.push((name, Some(data)));
    S_OK
}

/// # Safety
///
/// `update`, `name` are null or valid as the API this stands in for
/// requires: a structure or block of its type, a NUL-terminated string, or
/// a buffer for its stated size, for the duration of the call.
unsafe extern "system" fn delete_blob(
    _: This,
    update: *mut Update,
    name: *const c_char,
) -> HRESULT {
    // SAFETY: as in `write_blob`.
    match (unsafe { update.as_mut() }, text(name)) {
        (Some(update), Some(name)) if !name.is_empty() => {
            update.changes.push((name, None));
            S_OK
        }
        _ => E_INVALIDARG,
    }
}

/// # Safety
///
/// `update`, `block` are null or valid as the API this stands in for
/// requires: a structure or block of its type, a NUL-terminated string, or
/// a buffer for its stated size, for the duration of the call.
unsafe extern "system" fn submit_async(
    _: This,
    update: *const Update,
    block: *mut AsyncBlock,
) -> HRESULT {
    // SAFETY: as this function's contract gives them.
    let hr = unsafe { get(update) }.map_or(E_INVALIDARG, Update::submit);
    // SAFETY: the pointers passed on are under this function's own contract.
    unsafe { xasync::finished(block, (&raw const SUBMIT).cast(), hr) }
}
/// # Safety
///
/// `block` is null or valid as the API this stands in for requires: a
/// structure or block of its type, a NUL-terminated string, or a buffer for
/// its stated size, for the duration of the call.
unsafe extern "system" fn submit_result(_: This, block: *mut AsyncBlock) -> HRESULT {
    let none = std::ptr::null_mut();
    // SAFETY: the pointers passed on are under this function's own contract.
    unsafe { xasync::XAsyncGetResult(block, (&raw const SUBMIT).cast(), 0, none, none.cast()) }
}

static VTABLE: Methods<30> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    missing::<3> as *const (),         // 0x18 initialize (blocking)
    initialize_async as *const (),     // 0x20
    initialize_result as *const (),    // 0x28
    close_provider as *const (),       // 0x30
    missing::<7> as *const (),         // 0x38 remaining quota (blocking)
    quota_async as *const (),          // 0x40
    quota_result as *const (),         // 0x48
    missing::<10> as *const (),        // 0x50 delete (blocking)
    delete_async as *const (),         // 0x58
    delete_result as *const (),        // 0x60
    container_info as *const (),       // 0x68
    enumerate_containers as *const (), // 0x70
    missing::<15> as *const (),        // 0x78 by name prefix
    create_container as *const (),     // 0x80
    close_container as *const (),      // 0x88
    missing::<18> as *const (),        // 0x90 all blob info
    enumerate_blobs as *const (),      // 0x98 by name prefix
    read_blobs as *const (),           // 0xa0
    read_async as *const (),           // 0xa8
    read_result as *const (),          // 0xb0
    create_update as *const (),        // 0xb8
    close_update as *const (),         // 0xc0
    write_blob as *const (),           // 0xc8
    delete_blob as *const (),          // 0xd0
    missing::<27> as *const (),        // 0xd8 submit (blocking)
    submit_async as *const (),         // 0xe0
    submit_result as *const (),        // 0xe8
]);

#[cfg(test)]
mod tests;

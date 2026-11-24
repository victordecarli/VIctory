//! Virtual drives over host folders: on the console, title storage is a
//! drive of its own, and titles build paths on that assumption.
//!
//! Every title import that takes a path is bound here. A path on a virtual
//! drive (`X:\..`, `X:/..`, `X:`, `\\?\X:\..`) is rewritten to the host
//! folder behind it and handed to the host's own function; any other path
//! goes to the host unchanged. A rewritten narrow path goes to the host's
//! wide twin, so host folders outside the ANSI code page work; never to our
//! wide wrapper, which would rewrite the host path a second time when its
//! drive letter is mounted too. Letters are chosen among those the host
//! does not use, so no host drive is hidden.
#![allow(
    non_snake_case,
    reason = "exports carry the exact API names (`CreateFileW`) that `hle_export!` binds imports by"
)]
#![allow(clippy::too_many_arguments, reason = "the signature is fixed by the API title code calls")]

use std::ffi::{CStr, c_char, c_void};
use std::os::windows::ffi::OsStrExt;
use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use windows_sys::Win32::Foundation::{ERROR_PROC_NOT_FOUND, INVALID_HANDLE_VALUE, SetLastError};
use windows_sys::Win32::Globalization::{CP_ACP, MultiByteToWideChar};
use windows_sys::Win32::Storage::FileSystem::GetLogicalDrives;

use crate::{HleExport, HleModule, hle_export};

/// The host's own function: what the import would have been bound to, or
/// `None` if the host lacks it. Looked up once per call site; the caller
/// returns its API's failure for `None` (see [`host_function`]).
macro_rules! try_host {
    ($dll:literal, $name:literal: $ty:ty) => {{
        static ADDRESS: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
        let address = *ADDRESS.get_or_init(|| $crate::modules::files::host_function($dll, $name));
        address.map(|address| {
            // SAFETY: `address` is the host's export of `$dll!$name`, whose
            // documented signature is `$ty`.
            unsafe { std::mem::transmute::<usize, $ty>(address) }
        })
    }};
}
pub(crate) use try_host;

/// The host's `dll!name`; `None` if the host lacks it, logged the first
/// time for each export.
pub(crate) fn host_function(dll: &'static str, name: &'static str) -> Option<usize> {
    let address = crate::host::function(dll, name);
    if address.is_none() {
        static MISSING: Mutex<Vec<(&str, &str)>> = Mutex::new(Vec::new());
        let mut missing = MISSING.lock().unwrap_or_else(|e| e.into_inner());
        if !missing.contains(&(dll, name)) {
            tracing::error!("the host lacks {dll}!{name}; calls to it fail");
            missing.push((dll, name));
        }
    }
    address
}

/// Mounted drives: upper-case letter and host root (UTF-16, no trailing
/// separator, no NUL).
static DRIVES: Mutex<Vec<(u16, Vec<u16>)>> = Mutex::new(Vec::new());

fn drives() -> MutexGuard<'static, Vec<(u16, Vec<u16>)>> {
    DRIVES.lock().unwrap_or_else(|e| e.into_inner())
}

/// Gives the host folder `root` a drive letter (the same one if it already
/// has one): the highest letter neither the host nor another mount uses.
pub fn mount(root: &Path) -> Option<char> {
    let wide = root_wide(root);
    let mut drives = drives();
    if let Some(&(letter, _)) = drives.iter().find(|(_, r)| *r == wide) {
        return char::from_u32(letter.into());
    }
    // SAFETY: no arguments.
    let host = unsafe { GetLogicalDrives() };
    let letter = (b'D'..=b'Z').rev().find(|&l| {
        host >> (l - b'A') & 1 == 0 && !drives.iter().any(|(m, _)| *m == u16::from(l))
    })?;
    tracing::info!(drive = %char::from(letter), root = %root.display(), "virtual drive");
    drives.push((letter.into(), wide));
    Some(letter.into())
}

/// Gives the host folder `root` a drive letter the console fixes (titles
/// open their package as `G:\..` and temporary files as `T:\..`). A host
/// drive of that letter is hidden from the title.
pub fn mount_at(letter: char, root: &Path) {
    let letter = letter.to_ascii_uppercase();
    let Some(index) = u32::from(letter).checked_sub(u32::from('A')).filter(|i| *i < 26) else {
        return;
    };
    // SAFETY: no arguments.
    let hides_host_drive = unsafe { GetLogicalDrives() } >> index & 1 != 0;
    tracing::info!(drive = %letter, root = %root.display(), hides_host_drive, "virtual drive");
    let mut drives = drives();
    drives.retain(|(l, _)| u32::from(*l) != u32::from(letter));
    drives.push((letter as u16, root_wide(root)));
}

/// `root` as UTF-16, absolute, without trailing separators (a drive root
/// keeps its own).
fn root_wide(root: &Path) -> Vec<u16> {
    let root = std::path::absolute(root).unwrap_or_else(|_| root.to_owned());
    let mut wide: Vec<u16> = root.as_os_str().encode_wide().collect();
    while wide.len() > 3 && wide.last().is_some_and(|c| SEPARATORS.contains(c)) {
        wide.pop();
    }
    wide
}

const SEPARATORS: [u16; 2] = [b'\\' as u16, b'/' as u16];
const VERBATIM: [u16; 4] = [b'\\' as u16, b'\\' as u16, b'?' as u16, b'\\' as u16];

/// Host paths from this length on take the `\\?\` form, which lifts the
/// host's MAX_PATH limit (248 characters for a directory, 260 for a file).
const LONG: usize = 248;

/// `root` and the title's `tail` as a `\\?\` path. That form turns off the
/// host's own clean-up, so `/` becomes `\` and `.` and `..` are resolved
/// here (`..` stops at the root).
fn verbatim(root: &[u16], tail: &[u16]) -> Vec<u16> {
    let backslash = u16::from(b'\\');
    let mut out = if root.starts_with(&VERBATIM) {
        root.to_vec()
    } else if let Some(unc) = root.strip_prefix(&[backslash; 2][..]) {
        let mut out = VERBATIM.to_vec();
        out.extend("UNC\\".encode_utf16());
        out.extend_from_slice(unc);
        out
    } else {
        [&VERBATIM[..], root].concat()
    };
    while out.last() == Some(&backslash) {
        out.pop();
    }
    let dot = u16::from(b'.');
    let mut parts: Vec<&[u16]> = Vec::new();
    for part in tail.split(|c| SEPARATORS.contains(c)) {
        match part {
            [] => {}
            [d] if *d == dot => {}
            [d, e] if [*d, *e] == [dot; 2] => drop(parts.pop()),
            _ => parts.push(part),
        }
    }
    for part in parts {
        out.push(backslash);
        out.extend_from_slice(part);
    }
    out
}

/// The host path, NUL-terminated, for `path` (without its NUL) if it is on
/// a virtual drive.
pub(crate) fn translate(path: &[u16]) -> Option<Vec<u16>> {
    let (prefix, rest) = match path.strip_prefix(&VERBATIM[..]) {
        Some(rest) => (&VERBATIM[..], rest),
        None => (&[][..], path),
    };
    let &[letter, colon, ..] = rest else {
        return None;
    };
    if colon != u16::from(b':') {
        return None;
    }
    let letter = u8::try_from(letter).ok().filter(u8::is_ascii_alphabetic)?;
    let letter = u16::from(letter.to_ascii_uppercase());
    let drives = drives();
    let (_, root) = drives.iter().find(|(l, _)| *l == letter)?;
    let tail = &rest[2..];
    if prefix.is_empty() && root.len() + tail.len() >= LONG {
        let mut out = verbatim(root, tail);
        out.push(0);
        return Some(out);
    }
    let mut out = prefix.to_vec();
    out.extend_from_slice(root);
    // `X:` and `X:name` are relative to the root.
    if !tail.first().is_some_and(|c| SEPARATORS.contains(c)) {
        out.push(b'\\'.into());
    }
    let verbatim = !prefix.is_empty();
    out.extend(tail.iter().map(
        |&c| {
            if verbatim && c == u16::from(b'/') { b'\\'.into() } else { c }
        },
    ));
    out.push(0);
    Some(out)
}

/// A title's wide path: the host path when it is on a virtual drive, else
/// the title's own pointer.
struct Wide(*const u16, Option<Vec<u16>>);

impl Wide {
    fn ptr(&self) -> *const u16 {
        self.1.as_ref().map_or(self.0, |p| p.as_ptr())
    }
}

/// # Safety
///
/// `path` is null or a NUL-terminated UTF-16 string valid for reads.
unsafe fn wide(path: *const u16) -> Wide {
    if path.is_null() {
        return Wide(path, None);
    }
    // SAFETY: `path` is NUL-terminated, per this function's contract.
    let path_slice = unsafe {
        let len = (0..).take_while(|&i| *path.add(i) != 0).count();
        std::slice::from_raw_parts(path, len)
    };
    let host = translate(path_slice);
    trace(path_slice, host.as_deref());
    Wide(path, host)
}

/// Every title path at debug level, with the host path if rewritten.
fn trace(path: &[u16], host: Option<&[u16]>) {
    tracing::debug!(
        path = %String::from_utf16_lossy(path),
        host = host.map(|h| String::from_utf16_lossy(&h[..h.len() - 1])),
        "title path"
    );
}

/// A narrow path (ANSI code page, as the host decodes it) on a virtual
/// drive, as a wide host path.
///
/// # Safety
///
/// `path` is null or a NUL-terminated string valid for reads.
unsafe fn narrow(path: *const c_char) -> Option<Vec<u16>> {
    if path.is_null() {
        return None;
    }
    // SAFETY: `path` is NUL-terminated, per this function's contract.
    let bytes = unsafe { CStr::from_ptr(path) }.to_bytes();
    // The drive part is ASCII in every code page; decode only candidates.
    match bytes.strip_prefix(br"\\?\").unwrap_or(bytes) {
        [letter, b':', ..] if letter.is_ascii_alphabetic() => {}
        _ => {
            tracing::debug!(path = %String::from_utf8_lossy(bytes), "title path");
            return None;
        }
    }
    let len = i32::try_from(bytes.len()).ok()?;
    // SAFETY: sizing call, then a call with room for `n` units.
    let wide = unsafe {
        let n = MultiByteToWideChar(CP_ACP, 0, bytes.as_ptr(), len, std::ptr::null_mut(), 0);
        let mut wide = vec![0u16; usize::try_from(n).ok()?];
        MultiByteToWideChar(CP_ACP, 0, bytes.as_ptr(), len, wide.as_mut_ptr(), n);
        wide
    };
    let host = translate(&wide);
    trace(&wide, host.as_deref());
    host
}

/// An ASCII mode string (`"rb"`, `"w, ccs=UTF-8"`) for a wide CRT call.
///
/// # Safety
///
/// `mode` is null or a NUL-terminated string valid for reads.
unsafe fn widen(mode: *const c_char) -> Vec<u16> {
    if mode.is_null() {
        return vec![0];
    }
    // SAFETY: `mode` is NUL-terminated, per this function's contract.
    let bytes = unsafe { CStr::from_ptr(mode) }.to_bytes();
    bytes.iter().map(|&b| u16::from(b)).chain([0]).collect()
}

type Handle = *mut c_void;
type File = *mut c_void;

/// `ENOSYS` in the Microsoft C runtime: function not supported.
const ENOSYS: i32 = 40;
const INVALID_FILE_ATTRIBUTES: u32 = u32::MAX;

/// A Win32 call's failure when the host lacks the function behind it:
/// `ERROR_PROC_NOT_FOUND` as the thread's last error, then `value`.
fn win32_failure<T>(value: T) -> T {
    // SAFETY: no pointers.
    unsafe { SetLastError(ERROR_PROC_NOT_FOUND) };
    value
}

/// A C runtime call's failure when the host lacks the function behind it:
/// `ENOSYS` in `errno` (if the host's CRT has one to set), then `value`.
fn crt_failure<T>(value: T) -> T {
    if let Some(errno) = try_host!("ucrtbase.dll", "_errno": extern "C" fn() -> *mut i32) {
        let slot = errno();
        if !slot.is_null() {
            // SAFETY: `_errno` returns the calling thread's `errno`, checked
            // non-null; the title's CRT imports bind to this same ucrtbase.
            unsafe { slot.write(ENOSYS) };
        }
    }
    value
}

// kernel32 and its file API sets.

/// # Safety
///
/// `path` is null or a NUL-terminated UTF-16 string valid for reads. The
/// rest go to the host unchanged, as it requires.
pub unsafe extern "system" fn CreateFileW(
    path: *const u16,
    access: u32,
    share: u32,
    security: *const c_void,
    disposition: u32,
    flags: u32,
    template: Handle,
) -> Handle {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    let p = unsafe { wide(path) };
    let Some(create) = try_host!("kernel32.dll", "CreateFileW": extern "system" fn(*const u16, u32, u32, *const c_void, u32, u32, Handle) -> Handle)
    else {
        return win32_failure(INVALID_HANDLE_VALUE);
    };
    create(
        p.ptr(),
        access,
        share,
        security,
        disposition,
        flags,
        template,
    )
}

/// # Safety
///
/// `path` is null or a NUL-terminated string valid for reads. The rest go
/// to the host unchanged, as it requires.
pub unsafe extern "system" fn CreateFileA(
    path: *const c_char,
    access: u32,
    share: u32,
    security: *const c_void,
    disposition: u32,
    flags: u32,
    template: Handle,
) -> Handle {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    match unsafe { narrow(path) } {
        Some(p) => {
            let Some(create) = try_host!("kernel32.dll", "CreateFileW": extern "system" fn(*const u16, u32, u32, *const c_void, u32, u32, Handle) -> Handle)
            else {
                return win32_failure(INVALID_HANDLE_VALUE);
            };
            create(
                p.as_ptr(),
                access,
                share,
                security,
                disposition,
                flags,
                template,
            )
        }
        None => {
            let Some(create) = try_host!("kernel32.dll", "CreateFileA": extern "system" fn(*const c_char, u32, u32, *const c_void, u32, u32, Handle) -> Handle)
            else {
                return win32_failure(INVALID_HANDLE_VALUE);
            };
            create(path, access, share, security, disposition, flags, template)
        }
    }
}

/// Used by Bink.
///
/// # Safety
///
/// `path` is null or a NUL-terminated UTF-16 string valid for reads. The
/// rest go to the host unchanged, as it requires.
pub unsafe extern "system" fn CreateFile2(
    path: *const u16,
    access: u32,
    share: u32,
    disposition: u32,
    params: *const c_void,
) -> Handle {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    let p = unsafe { wide(path) };
    let Some(create) = try_host!("kernel32.dll", "CreateFile2": extern "system" fn(*const u16, u32, u32, u32, *const c_void) -> Handle)
    else {
        return win32_failure(INVALID_HANDLE_VALUE);
    };
    create(p.ptr(), access, share, disposition, params)
}

/// # Safety
///
/// `path` is null or a NUL-terminated string valid for reads. The rest go
/// to the host unchanged, as it requires.
pub unsafe extern "system" fn DeleteFileA(path: *const c_char) -> i32 {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    match unsafe { narrow(path) } {
        Some(p) => {
            match try_host!("kernel32.dll", "DeleteFileW": extern "system" fn(*const u16) -> i32) {
                Some(delete) => delete(p.as_ptr()),
                None => win32_failure(0),
            }
        }
        None => {
            match try_host!("kernel32.dll", "DeleteFileA": extern "system" fn(*const c_char) -> i32)
            {
                Some(delete) => delete(path),
                None => win32_failure(0),
            }
        }
    }
}

/// # Safety
///
/// `path` is null or a NUL-terminated UTF-16 string valid for reads. The
/// rest go to the host unchanged, as it requires.
pub unsafe extern "system" fn DeleteFileW(path: *const u16) -> i32 {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    let p = unsafe { wide(path) };
    match try_host!("kernel32.dll", "DeleteFileW": extern "system" fn(*const u16) -> i32) {
        Some(delete) => delete(p.ptr()),
        None => win32_failure(0),
    }
}

/// # Safety
///
/// `path` is null or a NUL-terminated UTF-16 string valid for reads. The
/// rest go to the host unchanged, as it requires.
pub unsafe extern "system" fn RemoveDirectoryW(path: *const u16) -> i32 {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    let p = unsafe { wide(path) };
    match try_host!("kernel32.dll", "RemoveDirectoryW": extern "system" fn(*const u16) -> i32) {
        Some(remove_directory) => remove_directory(p.ptr()),
        None => win32_failure(0),
    }
}

/// # Safety
///
/// `path` is null or a NUL-terminated UTF-16 string valid for reads. The
/// rest go to the host unchanged, as it requires.
pub unsafe extern "system" fn SetFileAttributesW(path: *const u16, attributes: u32) -> i32 {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    let p = unsafe { wide(path) };
    match try_host!("kernel32.dll", "SetFileAttributesW": extern "system" fn(*const u16, u32) -> i32)
    {
        Some(set) => set(p.ptr(), attributes),
        None => win32_failure(0),
    }
}

/// # Safety
///
/// `path` is null or a NUL-terminated string valid for reads. The rest go
/// to the host unchanged, as it requires.
pub unsafe extern "system" fn GetFileAttributesA(path: *const c_char) -> u32 {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    match unsafe { narrow(path) } {
        Some(p) => {
            match try_host!("kernel32.dll", "GetFileAttributesW": extern "system" fn(*const u16) -> u32)
            {
                Some(get) => get(p.as_ptr()),
                None => win32_failure(INVALID_FILE_ATTRIBUTES),
            }
        }
        None => {
            match try_host!("kernel32.dll", "GetFileAttributesA": extern "system" fn(*const c_char) -> u32)
            {
                Some(get) => get(path),
                None => win32_failure(INVALID_FILE_ATTRIBUTES),
            }
        }
    }
}

/// # Safety
///
/// `path` is null or a NUL-terminated UTF-16 string valid for reads. The
/// rest go to the host unchanged, as it requires.
pub unsafe extern "system" fn CreateDirectoryW(path: *const u16, security: *const c_void) -> i32 {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    let p = unsafe { wide(path) };
    match try_host!("kernel32.dll", "CreateDirectoryW": extern "system" fn(*const u16, *const c_void) -> i32)
    {
        Some(create) => create(p.ptr(), security),
        None => win32_failure(0),
    }
}

/// # Safety
///
/// `path` is null or a NUL-terminated UTF-16 string valid for reads. The
/// rest go to the host unchanged, as it requires.
pub unsafe extern "system" fn GetFileAttributesExW(
    path: *const u16,
    level: i32,
    info: *mut c_void,
) -> i32 {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    let p = unsafe { wide(path) };
    match try_host!("kernel32.dll", "GetFileAttributesExW": extern "system" fn(*const u16, i32, *mut c_void) -> i32)
    {
        Some(get) => get(p.ptr(), level, info),
        None => win32_failure(0),
    }
}

/// # Safety
///
/// `path` is null or a NUL-terminated UTF-16 string valid for reads. The
/// rest go to the host unchanged, as it requires.
pub unsafe extern "system" fn GetFileAttributesW(path: *const u16) -> u32 {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    let p = unsafe { wide(path) };
    match try_host!("kernel32.dll", "GetFileAttributesW": extern "system" fn(*const u16) -> u32) {
        Some(get) => get(p.ptr()),
        None => win32_failure(INVALID_FILE_ATTRIBUTES),
    }
}

/// # Safety
///
/// `path` is null or a NUL-terminated UTF-16 string valid for reads. The
/// rest go to the host unchanged, as it requires.
pub unsafe extern "system" fn GetDiskFreeSpaceExW(
    path: *const u16,
    available: *mut u64,
    total: *mut u64,
    free: *mut u64,
) -> i32 {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    let p = unsafe { wide(path) };
    match try_host!("kernel32.dll", "GetDiskFreeSpaceExW": extern "system" fn(*const u16, *mut u64, *mut u64, *mut u64) -> i32)
    {
        Some(get) => get(p.ptr(), available, total, free),
        None => win32_failure(0),
    }
}

/// Wants a volume root: a virtual drive answers for the host volume its
/// folder is on.
///
/// # Safety
///
/// `root` is null or a NUL-terminated UTF-16 string valid for reads. The
/// rest go to the host unchanged, as it requires.
pub unsafe extern "system" fn GetDiskFreeSpaceW(
    root: *const u16,
    sectors_per_cluster: *mut u32,
    bytes_per_sector: *mut u32,
    free_clusters: *mut u32,
    clusters: *mut u32,
) -> i32 {
    // SAFETY: `root` is null or NUL-terminated, per this function's contract.
    let mut p = unsafe { wide(root) };
    if let Some(host) = &mut p.1
        && let [letter, colon, sep, ..] = host[..]
        && colon == u16::from(b':')
        && SEPARATORS.contains(&sep)
    {
        *host = vec![letter, colon, sep, 0];
    }
    let Some(get) = try_host!("kernel32.dll", "GetDiskFreeSpaceW": extern "system" fn(*const u16, *mut u32, *mut u32, *mut u32, *mut u32) -> i32)
    else {
        return win32_failure(0);
    };
    get(
        p.ptr(),
        sectors_per_cluster,
        bytes_per_sector,
        free_clusters,
        clusters,
    )
}

/// # Safety
///
/// `path` is null or a NUL-terminated UTF-16 string valid for reads. The
/// rest go to the host unchanged, as it requires.
pub unsafe extern "system" fn FindFirstFileW(path: *const u16, data: *mut c_void) -> Handle {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    let p = unsafe { wide(path) };
    match try_host!("kernel32.dll", "FindFirstFileW": extern "system" fn(*const u16, *mut c_void) -> Handle)
    {
        Some(find) => find(p.ptr(), data),
        None => win32_failure(INVALID_HANDLE_VALUE),
    }
}

/// # Safety
///
/// `path` is null or a NUL-terminated UTF-16 string valid for reads. The
/// rest go to the host unchanged, as it requires.
pub unsafe extern "system" fn FindFirstFileExW(
    path: *const u16,
    level: i32,
    data: *mut c_void,
    search: i32,
    filter: *mut c_void,
    flags: u32,
) -> Handle {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    let p = unsafe { wide(path) };
    let Some(find) = try_host!("kernel32.dll", "FindFirstFileExW": extern "system" fn(*const u16, i32, *mut c_void, i32, *mut c_void, u32) -> Handle)
    else {
        return win32_failure(INVALID_HANDLE_VALUE);
    };
    find(p.ptr(), level, data, search, filter, flags)
}

/// # Safety
///
/// `from` is null or a NUL-terminated UTF-16 string valid for reads. `to`
/// is null or a NUL-terminated UTF-16 string valid for reads. The rest go
/// to the host unchanged, as it requires.
pub unsafe extern "system" fn MoveFileExW(from: *const u16, to: *const u16, flags: u32) -> i32 {
    // SAFETY: both are null or NUL-terminated, per this function's contract.
    let (from, to) = unsafe { (wide(from), wide(to)) };
    match try_host!("kernel32.dll", "MoveFileExW": extern "system" fn(*const u16, *const u16, u32) -> i32)
    {
        Some(move_file) => move_file(from.ptr(), to.ptr(), flags),
        None => win32_failure(0),
    }
}

/// # Safety
///
/// `path` is null or a NUL-terminated string valid for reads. The rest go
/// to the host unchanged, as it requires.
pub unsafe extern "system" fn SetCurrentDirectoryA(path: *const c_char) -> i32 {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    match unsafe { narrow(path) } {
        Some(p) => {
            match try_host!("kernel32.dll", "SetCurrentDirectoryW": extern "system" fn(*const u16) -> i32)
            {
                Some(set) => set(p.as_ptr()),
                None => win32_failure(0),
            }
        }
        None => {
            match try_host!("kernel32.dll", "SetCurrentDirectoryA": extern "system" fn(*const c_char) -> i32)
            {
                Some(set) => set(path),
                None => win32_failure(0),
            }
        }
    }
}

// The C runtime (ucrtbase, which the title's CRT imports bind to).

/// # Safety
///
/// `path` is null or a NUL-terminated UTF-16 string valid for reads. `mode`
/// is a NUL-terminated UTF-16 string valid for reads. The rest go to the
/// host unchanged, as it requires.
pub unsafe extern "C" fn _wfopen(path: *const u16, mode: *const u16) -> File {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    let p = unsafe { wide(path) };
    match try_host!("ucrtbase.dll", "_wfopen": extern "C" fn(*const u16, *const u16) -> File) {
        Some(open) => open(p.ptr(), mode),
        None => crt_failure(std::ptr::null_mut()),
    }
}

/// Without the host's `_wfopen_s`, `out` is left as the title passed it;
/// the error code is what callers check.
///
/// # Safety
///
/// `path` is null or a NUL-terminated UTF-16 string valid for reads. `mode`
/// is a NUL-terminated UTF-16 string valid for reads. `out` is valid for a
/// write of a stream pointer. The rest go to the host unchanged, as it
/// requires.
pub unsafe extern "C" fn _wfopen_s(out: *mut File, path: *const u16, mode: *const u16) -> i32 {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    let p = unsafe { wide(path) };
    match try_host!("ucrtbase.dll", "_wfopen_s": extern "C" fn(*mut File, *const u16, *const u16) -> i32)
    {
        Some(open) => open(out, p.ptr(), mode),
        None => crt_failure(ENOSYS),
    }
}

/// # Safety
///
/// `path` is null or a NUL-terminated UTF-16 string valid for reads. `mode`
/// is a NUL-terminated UTF-16 string valid for reads. The rest go to the
/// host unchanged, as it requires.
pub unsafe extern "C" fn _wfsopen(path: *const u16, mode: *const u16, share: i32) -> File {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    let p = unsafe { wide(path) };
    match try_host!("ucrtbase.dll", "_wfsopen": extern "C" fn(*const u16, *const u16, i32) -> File)
    {
        Some(open) => open(p.ptr(), mode, share),
        None => crt_failure(std::ptr::null_mut()),
    }
}

// Kernel reads into title memory. The GPU side leaves blocks of title GPU
// buffers read-only to see the title's next write, and the kernel's writes
// fail there instead of faulting into its handler; Grand Theft Auto VI's
// synchronous reads (title+0x313fc0) read straight into 4 KiB-aligned
// destinations and retry a failed `ReadFile` forever. So the destination
// is opened first.

/// # Safety
///
/// `buffer` is valid for writes of `len` bytes; `read` and `overlapped` are
/// as `ReadFile` requires. All go to the host unchanged.
pub unsafe extern "system" fn ReadFile(
    file: *mut c_void,
    buffer: *mut u8,
    len: u32,
    read: *mut u32,
    overlapped: *mut c_void,
) -> i32 {
    victory_gpu::kernel_will_write(buffer as usize, len as usize);
    match try_host!("kernel32.dll", "ReadFile": extern "system" fn(*mut c_void, *mut u8, u32, *mut u32, *mut c_void) -> i32)
    {
        Some(read_file) => read_file(file, buffer, len, read, overlapped),
        None => win32_failure(0),
    }
}

/// The host CRT reads large requests straight into the caller's buffer.
/// Without the host's `fread` nothing is read; the stream's error flag
/// cannot be set without it, so `errno` carries the failure.
///
/// # Safety
///
/// `buffer` is valid for writes of `size * count` bytes and `stream` is an
/// open stream; both go to the host unchanged.
pub unsafe extern "C" fn fread(buffer: *mut u8, size: usize, count: usize, stream: File) -> usize {
    victory_gpu::kernel_will_write(buffer as usize, size.saturating_mul(count));
    match try_host!("ucrtbase.dll", "fread": extern "C" fn(*mut u8, usize, usize, File) -> usize) {
        Some(read) => read(buffer, size, count, stream),
        None => crt_failure(0),
    }
}

/// # Safety
///
/// `path` is null or a NUL-terminated string valid for reads. `mode` is a
/// NUL-terminated string valid for reads. The rest go to the host
/// unchanged, as it requires.
pub unsafe extern "C" fn fopen(path: *const c_char, mode: *const c_char) -> File {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    match unsafe { narrow(path) } {
        Some(p) => {
            match try_host!("ucrtbase.dll", "_wfopen": extern "C" fn(*const u16, *const u16) -> File)
            {
                // SAFETY: `mode` is NUL-terminated, per this function's contract.
                Some(open) => open(p.as_ptr(), unsafe { widen(mode) }.as_ptr()),
                None => crt_failure(std::ptr::null_mut()),
            }
        }
        None => {
            match try_host!("ucrtbase.dll", "fopen": extern "C" fn(*const c_char, *const c_char) -> File)
            {
                Some(open) => open(path, mode),
                None => crt_failure(std::ptr::null_mut()),
            }
        }
    }
}

/// See [`_wfopen_s`] for `out` when the host lacks the function.
///
/// # Safety
///
/// `path` is null or a NUL-terminated string valid for reads. `mode` is a
/// NUL-terminated string valid for reads. `out` is valid for a write of a
/// stream pointer. The rest go to the host unchanged, as it requires.
pub unsafe extern "C" fn fopen_s(out: *mut File, path: *const c_char, mode: *const c_char) -> i32 {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    match unsafe { narrow(path) } {
        Some(p) => {
            match try_host!("ucrtbase.dll", "_wfopen_s": extern "C" fn(*mut File, *const u16, *const u16) -> i32)
            {
                // SAFETY: `mode` is NUL-terminated, per this function's contract.
                Some(open) => open(out, p.as_ptr(), unsafe { widen(mode) }.as_ptr()),
                None => crt_failure(ENOSYS),
            }
        }
        None => {
            match try_host!("ucrtbase.dll", "fopen_s": extern "C" fn(*mut File, *const c_char, *const c_char) -> i32)
            {
                Some(open) => open(out, path, mode),
                None => crt_failure(ENOSYS),
            }
        }
    }
}

/// `_open(path, flags[, mode])` is variadic; on x64 the optional integer
/// travels in the same register either way, so it is passed on as is.
///
/// # Safety
///
/// `path` is null or a NUL-terminated string valid for reads. The rest go
/// to the host unchanged, as it requires.
pub unsafe extern "C" fn _open(path: *const c_char, flags: i32, mode: i32) -> i32 {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    match unsafe { narrow(path) } {
        Some(p) => {
            match try_host!("ucrtbase.dll", "_wopen": extern "C" fn(*const u16, i32, i32) -> i32) {
                Some(open) => open(p.as_ptr(), flags, mode),
                None => crt_failure(-1),
            }
        }
        None => {
            match try_host!("ucrtbase.dll", "_open": extern "C" fn(*const c_char, i32, i32) -> i32)
            {
                Some(open) => open(path, flags, mode),
                None => crt_failure(-1),
            }
        }
    }
}

/// See [`_open`]; used by libHttpClient.
///
/// # Safety
///
/// `path` is null or a NUL-terminated UTF-16 string valid for reads. The
/// rest go to the host unchanged, as it requires.
pub unsafe extern "C" fn _wopen(path: *const u16, flags: i32, mode: i32) -> i32 {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    let p = unsafe { wide(path) };
    match try_host!("ucrtbase.dll", "_wopen": extern "C" fn(*const u16, i32, i32) -> i32) {
        Some(open) => open(p.ptr(), flags, mode),
        None => crt_failure(-1),
    }
}

/// # Safety
///
/// `path` is null or a NUL-terminated UTF-16 string valid for reads. The
/// rest go to the host unchanged, as it requires.
pub unsafe extern "C" fn _wstat64(path: *const u16, stat: *mut c_void) -> i32 {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    let p = unsafe { wide(path) };
    match try_host!("ucrtbase.dll", "_wstat64": extern "C" fn(*const u16, *mut c_void) -> i32) {
        Some(stat64) => stat64(p.ptr(), stat),
        None => crt_failure(-1),
    }
}

/// # Safety
///
/// `path` is null or a NUL-terminated string valid for reads. The rest go
/// to the host unchanged, as it requires.
pub unsafe extern "C" fn _stat64i32(path: *const c_char, stat: *mut c_void) -> i32 {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    match unsafe { narrow(path) } {
        Some(p) => {
            match try_host!("ucrtbase.dll", "_wstat64i32": extern "C" fn(*const u16, *mut c_void) -> i32)
            {
                Some(stat64) => stat64(p.as_ptr(), stat),
                None => crt_failure(-1),
            }
        }
        None => {
            match try_host!("ucrtbase.dll", "_stat64i32": extern "C" fn(*const c_char, *mut c_void) -> i32)
            {
                Some(stat64) => stat64(path, stat),
                None => crt_failure(-1),
            }
        }
    }
}

/// # Safety
///
/// `path` is null or a NUL-terminated string valid for reads. The rest go
/// to the host unchanged, as it requires.
pub unsafe extern "C" fn remove(path: *const c_char) -> i32 {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    match unsafe { narrow(path) } {
        Some(p) => match try_host!("ucrtbase.dll", "_wremove": extern "C" fn(*const u16) -> i32) {
            Some(host_remove) => host_remove(p.as_ptr()),
            None => crt_failure(-1),
        },
        None => match try_host!("ucrtbase.dll", "remove": extern "C" fn(*const c_char) -> i32) {
            Some(host_remove) => host_remove(path),
            None => crt_failure(-1),
        },
    }
}

/// # Safety
///
/// `path` is null or a NUL-terminated UTF-16 string valid for reads. The
/// rest go to the host unchanged, as it requires.
pub unsafe extern "C" fn _wchmod(path: *const u16, mode: i32) -> i32 {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    let p = unsafe { wide(path) };
    match try_host!("ucrtbase.dll", "_wchmod": extern "C" fn(*const u16, i32) -> i32) {
        Some(chmod) => chmod(p.ptr(), mode),
        None => crt_failure(-1),
    }
}

/// # Safety
///
/// `path` is null or a NUL-terminated UTF-16 string valid for reads. The
/// rest go to the host unchanged, as it requires.
pub unsafe extern "C" fn _wutime64(path: *const u16, times: *mut c_void) -> i32 {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    let p = unsafe { wide(path) };
    match try_host!("ucrtbase.dll", "_wutime64": extern "C" fn(*const u16, *mut c_void) -> i32) {
        Some(utime) => utime(p.ptr(), times),
        None => crt_failure(-1),
    }
}

// The C++ library: `std::basic_filebuf::open` goes through `_Fiopen`, and
// a null stream from it makes `open` fail.

const FIOPEN: &str = "?_Fiopen@std@@YAPEAU_iobuf@@PEBDHH@Z";
const FIOPEN_WIDE: &str = "?_Fiopen@std@@YAPEAU_iobuf@@PEB_WHH@Z";

/// `_Fiopen(const wchar_t* path, ios_base::openmode, int protection)`.
///
/// # Safety
///
/// `path` is null or a NUL-terminated UTF-16 string valid for reads. The
/// rest go to the host unchanged, as it requires.
unsafe extern "C" fn fiopen_wide(path: *const u16, mode: i32, protection: i32) -> File {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    let p = unsafe { wide(path) };
    match try_host!("msvcp140.dll", "?_Fiopen@std@@YAPEAU_iobuf@@PEB_WHH@Z": extern "C" fn(*const u16, i32, i32) -> File)
    {
        Some(open) => open(p.ptr(), mode, protection),
        None => crt_failure(std::ptr::null_mut()),
    }
}

/// `_Fiopen(const char* path, ios_base::openmode, int protection)`.
///
/// # Safety
///
/// `path` is null or a NUL-terminated string valid for reads. The rest go
/// to the host unchanged, as it requires.
unsafe extern "C" fn fiopen(path: *const c_char, mode: i32, protection: i32) -> File {
    // SAFETY: `path` is null or NUL-terminated, per this function's contract.
    match unsafe { narrow(path) } {
        Some(p) => {
            match try_host!("msvcp140.dll", "?_Fiopen@std@@YAPEAU_iobuf@@PEB_WHH@Z": extern "C" fn(*const u16, i32, i32) -> File)
            {
                Some(open) => open(p.as_ptr(), mode, protection),
                None => crt_failure(std::ptr::null_mut()),
            }
        }
        None => {
            match try_host!("msvcp140.dll", "?_Fiopen@std@@YAPEAU_iobuf@@PEBDHH@Z": extern "C" fn(*const c_char, i32, i32) -> File)
            {
                Some(open) => open(path, mode, protection),
                None => crt_failure(std::ptr::null_mut()),
            }
        }
    }
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "files",
        dlls: &[
            "kernel32.dll",
            "kernelbase.dll",
            "api-ms-win-core-file-*",
            "api-ms-win-core-processenvironment-*",
            "ucrtbase.dll",
            "api-ms-win-crt-stdio-*",
            "api-ms-win-crt-filesystem-*",
            "api-ms-win-crt-time-*",
            "msvcp140.dll",
        ],
        exports: vec![
            hle_export!(ReadFile),
            hle_export!(fread),
            hle_export!(CreateFileW),
            hle_export!(CreateFileA),
            hle_export!(CreateFile2),
            hle_export!(DeleteFileA),
            hle_export!(DeleteFileW),
            hle_export!(RemoveDirectoryW),
            hle_export!(SetFileAttributesW),
            hle_export!(GetFileAttributesA),
            hle_export!(CreateDirectoryW),
            hle_export!(GetFileAttributesW),
            hle_export!(GetFileAttributesExW),
            hle_export!(GetDiskFreeSpaceExW),
            hle_export!(GetDiskFreeSpaceW),
            hle_export!(FindFirstFileW),
            hle_export!(FindFirstFileExW),
            hle_export!(MoveFileExW),
            hle_export!(SetCurrentDirectoryA),
            hle_export!(_wfopen),
            hle_export!(_wfopen_s),
            hle_export!(_wfsopen),
            hle_export!(fopen),
            hle_export!(fopen_s),
            hle_export!(_open),
            hle_export!(_wopen),
            hle_export!(_wstat64),
            hle_export!(_stat64i32),
            hle_export!(remove),
            hle_export!(_wchmod),
            hle_export!(_wutime64),
            HleExport { name: FIOPEN, address: fiopen as *const () as usize },
            HleExport { name: FIOPEN_WIDE, address: fiopen_wide as *const () as usize },
        ],
    }
}

#[cfg(test)]
mod tests;

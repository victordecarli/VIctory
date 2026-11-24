use super::*;

fn w(s: &str) -> Vec<u16> {
    s.encode_utf16().chain([0]).collect()
}

/// A host export that does not exist is `None`, not a panic, and the
/// failures the call sites return then carry their APIs' error codes.
#[test]
#[cfg(windows)]
fn a_missing_host_export_fails_the_call() {
    let missing = try_host!("kernel32.dll", "VictoryNoSuchExport": extern "system" fn() -> i32);
    assert!(missing.is_none());
    let missing = try_host!("victory-no-such.dll", "CreateFileW": extern "system" fn() -> i32);
    assert!(missing.is_none());

    assert_eq!(win32_failure(INVALID_HANDLE_VALUE), INVALID_HANDLE_VALUE);
    // SAFETY: no arguments.
    let error = unsafe { windows_sys::Win32::Foundation::GetLastError() };
    assert_eq!(error, ERROR_PROC_NOT_FOUND);

    assert_eq!(crt_failure(-1), -1);
    let errno = try_host!("ucrtbase.dll", "_errno": extern "C" fn() -> *mut i32)
        .expect("ucrtbase exports _errno")();
    // SAFETY: `_errno` returns the calling thread's `errno`.
    assert_eq!(unsafe { errno.read() }, ENOSYS);
}

#[test]
fn kernel_reads_reach_the_buffer() {
    use std::os::windows::io::AsRawHandle;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.bin");
    std::fs::write(&path, b"0123456789").unwrap();
    let file = std::fs::File::open(&path).unwrap();
    let (mut out, mut n) = ([0u8; 10], 0);
    let handle = file.as_raw_handle().cast();
    // SAFETY: `handle` is an open file, `out` holds 10 bytes and `n` is
    // a local to write.
    let read = unsafe { ReadFile(handle, out.as_mut_ptr(), 10, &mut n, std::ptr::null_mut()) };
    assert_ne!(read, 0);
    assert_eq!((&out, n), (b"0123456789", 10));
}

fn text(v: &[u16]) -> String {
    String::from_utf16_lossy(&v[..v.len() - 1])
}

#[test]
fn rewrites_only_paths_on_mounted_drives() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_string_lossy().into_owned();
    let letter = mount(dir.path()).unwrap();
    assert_eq!(mount(dir.path()), Some(letter), "mounting twice reuses the letter");
    let tr = |s: String| translate(&w(&s)[..s.encode_utf16().count()]).map(|v| text(&v));
    assert_eq!(tr(format!("{letter}:/a/b.txt")), Some(format!("{root}/a/b.txt")));
    let lower = letter.to_ascii_lowercase();
    assert_eq!(tr(format!("{lower}:\\a")), Some(format!("{root}\\a")));
    assert_eq!(tr(format!("{letter}:")), Some(format!("{root}\\")));
    assert_eq!(tr(format!(r"\\?\{letter}:/a/b")), Some(format!(r"\\?\{root}\a\b")));
    assert_eq!(tr(r"C:\Windows".into()), None);
    assert_eq!(tr("relative/path".into()), None);
    assert_eq!(tr(String::new()), None);

    // Fixed letters, as the console's package drive; a later mount of
    // the same letter replaces the earlier one.
    let package = tempfile::tempdir().unwrap();
    mount_at('g', dir.path());
    mount_at('G', package.path());
    assert_eq!(
        tr(r"G:\appdata0.rpf".into()),
        Some(format!(r"{}\appdata0.rpf", package.path().display()))
    );
    assert_ne!(mount(dir.path()), Some('G'), "free letters skip fixed ones");

    // Long ones take the `\\?\` form, cleaned up as the host would.
    let long = "d".repeat(250);
    assert_eq!(
        tr(format!(r"G:/a/./b/../{long}//f")),
        Some(format!(r"\\?\{}\a\{long}\f", package.path().display()))
    );
    assert_eq!(
        verbatim(&w(r"\\server\share\")[..15], &w("/x/../../y")[..10]),
        w(r"\\?\UNC\server\share\y")[..22].to_vec()
    );
}

#[test]
fn long_title_paths_work_on_the_host() {
    let dir = tempfile::tempdir().unwrap();
    let letter = mount(dir.path()).unwrap();
    let deep = format!("{letter}:/{}/{}", "a".repeat(200), "b".repeat(100));
    let null = std::ptr::null_mut();
    let first = w(&deep[..203]);
    // SAFETY: `w` ends each string with a NUL; a null security descriptor.
    assert_ne!(unsafe { CreateDirectoryW(first.as_ptr(), null) }, 0);
    // SAFETY: as above.
    assert_ne!(unsafe { CreateDirectoryW(w(&deep).as_ptr(), null) }, 0);
    let file = format!("{deep}/../{}/f.txt", "b".repeat(100));
    // SAFETY: NUL-terminated path and mode.
    let f = unsafe { _wfopen(w(&file).as_ptr(), w("wb").as_ptr()) };
    assert!(!f.is_null(), "{file}");
    try_host!("ucrtbase.dll", "fclose": extern "C" fn(File) -> i32)
        .expect("ucrtbase exports fclose")(f);
    // SAFETY: NUL-terminated paths.
    assert_ne!(unsafe { DeleteFileW(w(&format!("{deep}/f.txt")).as_ptr()) }, 0);
    // SAFETY: as above.
    assert_ne!(unsafe { RemoveDirectoryW(w(&deep).as_ptr()) }, 0);
}

#[test]
fn title_file_calls_land_in_the_host_folder() {
    let dir = tempfile::tempdir().unwrap();
    let letter = mount(dir.path()).unwrap();
    const GENERIC_WRITE: u32 = 0x4000_0000;
    const CREATE_ALWAYS: u32 = 2;
    let close = try_host!("kernel32.dll", "CloseHandle": extern "system" fn(Handle) -> i32)
        .expect("kernel32 exports CloseHandle");
    let fclose = try_host!("ucrtbase.dll", "fclose": extern "C" fn(File) -> i32)
        .expect("ucrtbase exports fclose");

    let path = w(&format!(r"{letter}:\created.bin"));
    let null = std::ptr::null_mut();
    // SAFETY: `path` is NUL-terminated; null security and template.
    let h = unsafe { CreateFileW(path.as_ptr(), GENERIC_WRITE, 0, null, CREATE_ALWAYS, 0, null) };
    assert_ne!(h as isize, -1);
    close(h);
    assert!(dir.path().join("created.bin").is_file());

    let path = format!("{letter}:/opened.txt\0");
    // SAFETY: NUL-terminated path and mode.
    let f = unsafe { fopen(path.as_ptr().cast(), c"wb".as_ptr()) };
    assert!(!f.is_null());
    fclose(f);
    assert!(dir.path().join("opened.txt").is_file());

    let path = format!("{letter}:/stream.txt\0");
    const OUT: i32 = 2;
    // SAFETY: a NUL-terminated path.
    let f = unsafe { fiopen(path.as_ptr().cast(), OUT, 0x40) };
    assert!(!f.is_null());
    fclose(f);
    assert!(dir.path().join("stream.txt").is_file());

    // SAFETY: as above.
    assert_eq!(unsafe { remove(path.as_ptr().cast()) }, 0);
    assert!(!dir.path().join("stream.txt").exists());
    // SAFETY: a NUL-terminated path.
    let f = unsafe { fiopen_wide(w(&format!("{letter}:/wide.txt")).as_ptr(), OUT, 0x40) };
    assert!(!f.is_null());
    fclose(f);
    assert!(dir.path().join("wide.txt").is_file());

    const OPEN_EXISTING: u32 = 3;
    let path = w(&format!(r"{letter}:\created.bin"));
    // SAFETY: `path` is NUL-terminated; null parameters.
    let h = unsafe { CreateFile2(path.as_ptr(), GENERIC_WRITE, 0, OPEN_EXISTING, null) };
    assert_ne!(h as isize, -1);
    close(h);
    const INVALID_FILE_ATTRIBUTES: u32 = u32::MAX;
    // SAFETY: as above.
    assert_ne!(unsafe { GetFileAttributesW(path.as_ptr()) }, INVALID_FILE_ATTRIBUTES);

    const O_WRONLY: i32 = 1;
    const O_CREAT: i32 = 0x100;
    const S_IWRITE: i32 = 0x80;
    let path = format!("{letter}:/low.bin\0");
    // SAFETY: a NUL-terminated path.
    let fd = unsafe { _open(path.as_ptr().cast(), O_WRONLY | O_CREAT, S_IWRITE) };
    assert!(fd >= 0);
    try_host!("ucrtbase.dll", "_close": extern "C" fn(i32) -> i32)
        .expect("ucrtbase exports _close")(fd);
    assert!(dir.path().join("low.bin").is_file());

    // Other paths pass through untouched.
    let host_path = format!("{}\0", dir.path().join("plain.txt").display());
    // SAFETY: NUL-terminated path and mode.
    let f = unsafe { fopen(host_path.as_ptr().cast(), c"wb".as_ptr()) };
    assert!(!f.is_null());
    fclose(f);
    assert!(dir.path().join("plain.txt").is_file());
}

/// A narrow path is rewritten once, as its wide twin is. The inner
/// drive's folder lies on the outer virtual drive, as the title folder
/// does when it sits on a host `G:` drive and is mounted as `G:`. The
/// outer letter is no host drive, so the host finds nothing there; a
/// narrow call that reaches the outer folder rewrote the path twice.
#[test]
fn narrow_paths_are_rewritten_once() {
    let dir = tempfile::tempdir().unwrap();
    let folder = dir.path().join("inner");
    std::fs::create_dir(&folder).unwrap();
    std::fs::write(folder.join("present.txt"), b"").unwrap();
    let outer = mount(dir.path()).unwrap();
    let inner = mount(Path::new(&format!(r"{outer}:\inner"))).unwrap();
    let path = |name: &str| format!("{inner}:/{name}\0");

    const INVALID_FILE_ATTRIBUTES: u32 = u32::MAX;
    // SAFETY: a NUL-terminated path.
    let wide = unsafe { GetFileAttributesW(w(&format!("{inner}:/present.txt")).as_ptr()) };
    assert_eq!(wide, INVALID_FILE_ATTRIBUTES);
    // SAFETY: as above, narrow.
    let narrow = unsafe { GetFileAttributesA(path("present.txt").as_ptr().cast()) };
    assert_eq!(narrow, wide, "GetFileAttributesA agrees with its wide twin");

    const GENERIC_WRITE: u32 = 0x4000_0000;
    const CREATE_ALWAYS: u32 = 2;
    let null = std::ptr::null_mut();
    // SAFETY: a NUL-terminated path; null security and template.
    let h = unsafe {
        CreateFileA(path("a.bin").as_ptr().cast(), GENERIC_WRITE, 0, null, CREATE_ALWAYS, 0, null)
    };
    assert_eq!(h as isize, -1);
    // SAFETY: NUL-terminated path and mode.
    assert!(unsafe { fopen(path("b.txt").as_ptr().cast(), c"wb".as_ptr()) }.is_null());
    let mut f: File = std::ptr::null_mut();
    // SAFETY: NUL-terminated path and mode; `f` is a local to write.
    let error = unsafe { fopen_s(&mut f, path("c.txt").as_ptr().cast(), c"wb".as_ptr()) };
    assert_ne!(error, 0);
    const OUT: i32 = 2;
    // SAFETY: a NUL-terminated path.
    assert!(unsafe { fiopen(path("d.txt").as_ptr().cast(), OUT, 0x40) }.is_null());

    // Nothing was made behind a path rewritten twice.
    let names: Vec<_> =
        std::fs::read_dir(&folder).unwrap().map(|e| e.unwrap().file_name()).collect();
    assert_eq!(names, ["present.txt"]);
}

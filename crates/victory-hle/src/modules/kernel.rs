//! Overrides of shared Windows APIs whose console meaning differs.
//!
//! Only the differences live here; everything else passes through to the
//! host (see `HostPolicy`).
#![allow(
    non_snake_case,
    reason = "exports carry the exact API names (`GetModuleHandleW`) that `hle_export!` binds imports by"
)]

use std::ffi::{CStr, c_void};

use crate::{HleModule, HostPolicy, hle_export, registry, runtime};

/// `GetModuleHandleW(NULL)` must return the title's image, not our exe;
/// title images and our console modules are found by name.
///
/// # Safety
///
/// `name` is null or points to a NUL-terminated UTF-16 string valid for
/// reads for the duration of the call; a name that is not ours is handed to
/// the host, which reads it too.
pub unsafe extern "system" fn GetModuleHandleW(name: *const u16) -> *mut c_void {
    if name.is_null() {
        return runtime::main_image().unwrap_or(0) as *mut c_void;
    }
    // SAFETY: `name` is non-null here, a NUL-terminated string per this
    // function's contract; the host reads the same string.
    unsafe { find_module(&wide(name), false).unwrap_or_else(|| host::get_module_handle_w(name)) }
}

/// See [`GetModuleHandleW`].
///
/// # Safety
///
/// `name` is null or points to a NUL-terminated string valid for reads for
/// the duration of the call; a name that is not ours is handed to the host,
/// which reads it too.
pub unsafe extern "system" fn GetModuleHandleA(name: *const u8) -> *mut c_void {
    if name.is_null() {
        return runtime::main_image().unwrap_or(0) as *mut c_void;
    }
    // SAFETY: `name` is non-null here, a NUL-terminated string per this
    // function's contract; the host reads the same string.
    unsafe { find_module(&narrow(name), false).unwrap_or_else(|| host::get_module_handle_a(name)) }
}

/// A NUL-terminated string from the title.
///
/// # Safety
///
/// `text` points to a NUL-terminated string valid for reads for the
/// duration of the call.
unsafe fn narrow(text: *const u8) -> String {
    // SAFETY: `text` is a NUL-terminated string, per this function's
    // contract.
    unsafe { CStr::from_ptr(text.cast()) }.to_string_lossy().into_owned()
}

/// See [`narrow`].
///
/// # Safety
///
/// `text` points to a NUL-terminated UTF-16 string valid for reads for the
/// duration of the call.
unsafe fn wide(text: *const u16) -> String {
    // SAFETY: `text` is a NUL-terminated string, per this function's
    // contract.
    unsafe {
        let len = (0..).take_while(|&i| *text.add(i) != 0).count();
        String::from_utf16_lossy(std::slice::from_raw_parts(text, len))
    }
}

/// A title image's path is its original file, not our cache copy;
/// `NULL` means the title's exe.
///
/// # Safety
///
/// `out` is valid for writes of `size` characters. A module that is not a
/// title image is handed to the host with the same buffer.
pub unsafe extern "system" fn GetModuleFileNameW(
    module: *mut c_void,
    out: *mut u16,
    size: u32,
) -> u32 {
    let Some(path) = title_path(module) else {
        // SAFETY: the same buffer, per this function's contract.
        return unsafe { host::get_module_file_name_w(module, out, size) };
    };
    let wide: Vec<u16> = path.encode_utf16().collect();
    // SAFETY: `out` holds `size` characters, per this function's contract.
    copy_truncated(&wide, unsafe { std::slice::from_raw_parts_mut(out, size as usize) })
}

/// See [`GetModuleFileNameW`].
///
/// # Safety
///
/// `out` is valid for writes of `size` bytes. A module that is not a title
/// image is handed to the host with the same buffer.
pub unsafe extern "system" fn GetModuleFileNameA(
    module: *mut c_void,
    out: *mut u8,
    size: u32,
) -> u32 {
    let Some(path) = title_path(module) else {
        // SAFETY: the same buffer, per this function's contract.
        return unsafe { host::get_module_file_name_a(module, out, size) };
    };
    // SAFETY: `out` holds `size` bytes, per this function's contract.
    copy_truncated(path.as_bytes(), unsafe { std::slice::from_raw_parts_mut(out, size as usize) })
}

fn title_path(module: *mut c_void) -> Option<String> {
    let base = if module.is_null() { runtime::main_image()? } else { module as usize };
    Some(runtime::image_path(base)?.to_string_lossy().into_owned())
}

/// `GetModuleFileName`'s contract: copy with a NUL; if it does not fit,
/// truncate to the buffer, report `ERROR_INSUFFICIENT_BUFFER` and return
/// the buffer size.
fn copy_truncated<T: Copy + Default>(text: &[T], out: &mut [T]) -> u32 {
    let Some(room) = out.len().checked_sub(1) else {
        host::set_last_error(ERROR_INSUFFICIENT_BUFFER);
        return 0;
    };
    let n = text.len().min(room);
    out[..n].copy_from_slice(&text[..n]);
    out[n] = T::default();
    if n < text.len() {
        host::set_last_error(ERROR_INSUFFICIENT_BUFFER);
        return out.len() as u32;
    }
    n as u32
}

const ERROR_INSUFFICIENT_BUFFER: u32 = 122;

/// Console-only DLLs we implement load as a pseudo handle (titles load
/// `xgameruntime.dll` this way), the title's DLLs as the images we map
/// (on first use if nothing imports them); anything else loads on the host.
///
/// # Safety
///
/// `name` is null or points to a NUL-terminated string valid for reads for
/// the duration of the call. `file` is null (reserved), as the API
/// requires; both go to the host for a library that is not ours.
pub unsafe extern "system" fn LoadLibraryExA(
    name: *const u8,
    file: *mut c_void,
    flags: u32,
) -> *mut c_void {
    // SAFETY: `name` is checked non-null, so a NUL-terminated string per
    // this function's contract; the host gets the same pointers.
    unsafe {
        if !name.is_null()
            && let Some(handle) = find_module(&narrow(name), runs_code(flags))
        {
            return handle;
        }
        host::load_library_ex_a(name, file, flags)
    }
}

/// See [`LoadLibraryExA`].
///
/// # Safety
///
/// `name` is null or points to a NUL-terminated UTF-16 string valid for
/// reads for the duration of the call. `file` is null (reserved), as the
/// API requires; both go to the host for a library that is not ours.
pub unsafe extern "system" fn LoadLibraryExW(
    name: *const u16,
    file: *mut c_void,
    flags: u32,
) -> *mut c_void {
    // SAFETY: `name` is checked non-null, so a NUL-terminated string per
    // this function's contract; the host gets the same pointers.
    unsafe {
        if !name.is_null()
            && let Some(handle) = find_module(&wide(name), runs_code(flags))
        {
            return handle;
        }
        host::load_library_ex_w(name, file, flags)
    }
}

/// False for loads as data or resources (`DONT_RESOLVE_DLL_REFERENCES`,
/// `LOAD_LIBRARY_AS_DATAFILE(_EXCLUSIVE)`, `AS_IMAGE_RESOURCE`): those run
/// no code, so a title DLL we have not mapped can go to the host.
fn runs_code(flags: u32) -> bool {
    flags & (0x1 | 0x2 | 0x20 | 0x40) == 0
}

/// # Safety
///
/// `name` is null or points to a NUL-terminated string valid for reads for
/// the duration of the call.
pub unsafe extern "system" fn LoadLibraryA(name: *const u8) -> *mut c_void {
    // SAFETY: `name` as this function's contract gives it; a null `file`.
    unsafe { LoadLibraryExA(name, std::ptr::null_mut(), 0) }
}

/// # Safety
///
/// `name` is null or points to a NUL-terminated UTF-16 string valid for
/// reads for the duration of the call.
pub unsafe extern "system" fn LoadLibraryW(name: *const u16) -> *mut c_void {
    // SAFETY: `name` as this function's contract gives it; a null `file`.
    unsafe { LoadLibraryExW(name, std::ptr::null_mut(), 0) }
}

/// Looks up our implementations behind a pseudo handle; real modules go to
/// the host.
///
/// # Safety
///
/// `name` is an ordinal (a value below 0x10000) or points to a
/// NUL-terminated string valid for reads for the duration of the call; the
/// host reads it too for a module that is not ours.
pub unsafe extern "system" fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void {
    // `NULL` is the process's exe, which for the title is its own image.
    let module =
        if module.is_null() { runtime::main_image().unwrap_or(0) as *mut c_void } else { module };
    let Some(m) = registry().iter().find(|m| handle(m) == module) else {
        // SAFETY: `name` is an ordinal or a NUL-terminated string, per this
        // function's contract; the host gets the same pointer.
        return unsafe {
            host_override(module, name).unwrap_or_else(|| host::get_proc_address(module, name))
        };
    };
    // Values below 0x10000 are ordinals; we export by name only.
    if (name as usize) < 0x1_0000 {
        tracing::warn!(module = m.name, ordinal = name as usize, "GetProcAddress by ordinal");
        return std::ptr::null_mut();
    }
    // SAFETY: not an ordinal, so a NUL-terminated name, per this function's
    // contract.
    let name = unsafe { CStr::from_ptr(name.cast()) }.to_string_lossy();
    match m.exports.iter().find(|e| e.name == name) {
        Some(e) => e.address as *mut c_void,
        None => {
            tracing::warn!(module = m.name, %name, "GetProcAddress: not implemented");
            std::ptr::null_mut()
        }
    }
}

/// What a host module's `name` (or ordinal) is bound to at import binding
/// when we override it: a title that looks a function up by name must not
/// get past our overrides (the network guard, path rewriting).
///
/// # Safety
///
/// `name` is an ordinal (a value below 0x10000) or points to a
/// NUL-terminated string valid for reads for the duration of the call.
unsafe fn host_override(module: *mut c_void, name: *const u8) -> Option<*mut c_void> {
    if module.is_null() || runtime::image_path(module as usize).is_some() {
        return None;
    }
    let symbol = if (name as usize) < 0x1_0000 {
        format!("#{}", name as usize)
    } else {
        // SAFETY: not an ordinal, so a NUL-terminated name, per this
        // function's contract.
        unsafe { narrow(name) }
    };
    let mut path = [0u16; 260];
    // SAFETY: `path` is a local buffer of `path.len()` characters.
    let len = unsafe { host::get_module_file_name_w(module, path.as_mut_ptr(), path.len() as u32) }
        as usize;
    let path = String::from_utf16_lossy(&path[..len.min(path.len())]);
    let file = path.rsplit('\\').next()?;
    crate::registry::lookup(file, &symbol).map(|address| address as *mut c_void)
}

/// Our pseudo modules and the title's images stay loaded: `LoadLibrary`
/// took no reference on them, so the host's count must not drop.
pub extern "system" fn FreeLibrary(module: *mut c_void) -> i32 {
    if registry().iter().any(|m| handle(m) == module)
        || runtime::image_path(module as usize).is_some()
    {
        return 1;
    }
    host::free_library(module)
}

/// Title images hear of thread starts and ends from us, not from the
/// Windows loader (`victory_loader::disable_thread_calls`).
pub extern "system" fn DisableThreadLibraryCalls(module: *mut c_void) -> i32 {
    host::disable_thread_library_calls(module)
}

/// Title debug output goes to our log (no debugger needs to be attached).
///
/// # Safety
///
/// `text` is null or points to a NUL-terminated string valid for reads for
/// the duration of the call.
pub unsafe extern "system" fn OutputDebugStringA(text: *const u8) {
    if !text.is_null() {
        // SAFETY: checked non-null; a NUL-terminated string, per this
        // function's contract.
        let text = unsafe { narrow(text) };
        host::note_title_output(text.trim());
        tracing::info!(target: "title", "{}", text.trim_end());
    }
}

/// See [`OutputDebugStringA`].
///
/// # Safety
///
/// `text` is null or points to a NUL-terminated UTF-16 string valid for
/// reads for the duration of the call.
pub unsafe extern "system" fn OutputDebugStringW(text: *const u16) {
    if !text.is_null() {
        // SAFETY: checked non-null; a NUL-terminated string, per this
        // function's contract.
        let text = unsafe { wide(text) };
        host::note_title_output(text.trim());
        tracing::info!(target: "title", "{}", text.trim_end());
    }
}

/// The CRT's `exit`, through which the title's entry point returns: logs
/// the code and the caller, then exits on the host.
pub extern "C" fn exit(code: i32) -> ! {
    tracing::warn!(code, from = %host::callers(), "title exits");
    host::exit(code)
}

/// The CRT's `abort` (also reached from `terminate`): logged like [`exit`].
pub extern "C" fn abort() -> ! {
    tracing::error!(from = %host::callers(), "title aborts");
    host::abort()
}

/// The CRT's `terminate` and the STL's invalid-argument exit: logged, then
/// the host's `abort`.
pub extern "C" fn terminate() -> ! {
    tracing::error!(from = %host::callers(), "title calls terminate");
    host::abort()
}

/// See [`terminate`].
pub extern "C" fn _invalid_parameter_noinfo_noreturn() -> ! {
    tracing::error!(from = %host::callers(), "title passed an invalid parameter");
    host::abort()
}

/// The title's unhandled-exception filter. Ours stands in front of it and
/// reports the crash first; the title still gets its own back.
static TITLE_FILTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub extern "system" fn SetUnhandledExceptionFilter(filter: usize) -> usize {
    host::set_unhandled_exception_filter(report_unhandled);
    TITLE_FILTER.swap(filter, std::sync::atomic::Ordering::AcqRel)
}

extern "system" fn report_unhandled(info: *const c_void) -> i32 {
    // SAFETY: the system calls this filter with its exception information.
    unsafe { host::report_unhandled(info) };
    match TITLE_FILTER.load(std::sync::atomic::Ordering::Acquire) {
        0 => 0, // EXCEPTION_CONTINUE_SEARCH
        filter => {
            type Filter = extern "system" fn(*const c_void) -> i32;
            // SAFETY: the title registered this filter.
            let filter: Filter = unsafe { std::mem::transmute(filter) };
            filter(info)
        }
    }
}

/// See [`exit`]; only the current process is logged.
pub extern "system" fn TerminateProcess(process: *mut c_void, code: u32) -> i32 {
    if process as isize == -1 {
        tracing::warn!(code, from = %host::callers(), "title terminates the process");
    }
    host::terminate_process(process, code)
}

/// The pseudo handle of one of our modules: its address, never a real image.
fn handle(m: &HleModule) -> *mut c_void {
    m as *const HleModule as *mut c_void
}

/// The title's images and our console modules by name; `None` leaves the
/// name to the host, which only DLLs `HostPolicy` passes through get (as at
/// import binding); others fail. With `load`, a title DLL not mapped yet is
/// mapped now.
fn find_module(path: &str, load: bool) -> Option<*mut c_void> {
    let file = path.rsplit(['\\', '/']).next().unwrap_or(path);
    // As the Windows loader does, a name without extension means a DLL.
    let file = if file.contains('.') { file.to_owned() } else { format!("{file}.dll") };
    // Loads go through the title DLL loader first: it waits for a DLL that
    // another thread is still initialising, as the Windows loader lock does.
    if load {
        match runtime::load_title_dll(&file) {
            Ok(Some(base)) => return Some(base as *mut c_void),
            Ok(None) => {}
            // Not the host: it would load the unpatched DLL from the title's folder.
            Err(error) => {
                tracing::error!(dll = file, %error, "mapping a title DLL failed");
                host::set_last_error(ERROR_MOD_NOT_FOUND);
                return Some(std::ptr::null_mut());
            }
        }
    }
    if let Some(base) = runtime::image_named(&file) {
        return Some(base as *mut c_void);
    }
    // DLLs the host has too are real host modules, whatever `--strict` says.
    let host_dll = HostPolicy::default().allows(&file);
    if !host_dll && let Some(m) = registry().iter().find(|m| m.serves(&file)) {
        if load {
            tracing::info!(dll = file, "loaded as a pseudo module");
        }
        return Some(handle(m));
    }
    if host_dll {
        return None;
    }
    // Any other host DLL (wininet, urlmon, ...) is beyond what we guard, and
    // the console does not have it.
    refuse(&file);
    host::set_last_error(ERROR_MOD_NOT_FOUND);
    Some(std::ptr::null_mut())
}

/// Logs a DLL name the title is refused, once.
fn refuse(file: &str) {
    static REFUSED: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let file = file.to_ascii_lowercase();
    let mut refused = REFUSED.lock().unwrap_or_else(|e| e.into_inner());
    if !refused.contains(&file) {
        tracing::warn!(dll = file, "a host DLL outside the host policy: refused");
        refused.push(file);
    }
}

const ERROR_MOD_NOT_FOUND: u32 = 126;

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "kernel",
        dlls: &[
            "kernel32.dll",
            "kernelbase.dll",
            "api-ms-win-core-libraryloader-*",
            "api-ms-win-core-processthreads-*",
            "api-ms-win-core-debug-*",
            "api-ms-win-core-errorhandling-*",
            "api-ms-win-crt-runtime-*",
        ],
        exports: vec![
            hle_export!(GetModuleHandleW),
            hle_export!(GetModuleHandleA),
            hle_export!(GetModuleFileNameW),
            hle_export!(GetModuleFileNameA),
            hle_export!(LoadLibraryExA),
            hle_export!(LoadLibraryExW),
            hle_export!(LoadLibraryA),
            hle_export!(LoadLibraryW),
            hle_export!(GetProcAddress),
            hle_export!(FreeLibrary),
            hle_export!(DisableThreadLibraryCalls),
            hle_export!(OutputDebugStringA),
            hle_export!(OutputDebugStringW),
            hle_export!(exit),
            hle_export!(abort),
            hle_export!(terminate),
            hle_export!(_invalid_parameter_noinfo_noreturn),
            hle_export!(TerminateProcess),
            hle_export!(SetUnhandledExceptionFilter),
        ],
    }
}

/// The return addresses above the caller, as `module+rva`, for logs.
#[cfg(feature = "diagnostics")]
pub(crate) fn callers() -> String {
    host::callers()
}

#[cfg(windows)]
mod host {
    use std::ffi::c_void;

    use windows_sys::Win32::Foundation::{FreeLibrary, SetLastError};
    use windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace;
    use windows_sys::Win32::System::LibraryLoader::{
        DisableThreadLibraryCalls, GetModuleFileNameA, GetModuleFileNameW, GetModuleHandleA,
        GetModuleHandleW, GetProcAddress, LoadLibraryExA, LoadLibraryExW,
    };
    use windows_sys::Win32::System::Threading::TerminateProcess;

    unsafe extern "C" {
        #[link_name = "exit"]
        fn crt_exit(code: i32) -> !;
        #[link_name = "abort"]
        fn crt_abort() -> !;
    }

    // SAFETY (the functions that call the host): forwarding the title's own
    // arguments to the host API of the same name, which needs what each
    // function's contract says.

    /// # Safety
    ///
    /// `name` is null or points to a NUL-terminated UTF-16 string valid for
    /// reads.
    pub(super) unsafe fn get_module_handle_w(name: *const u16) -> *mut c_void {
        // SAFETY: the pointers are as this function's contract gives them.
        unsafe { GetModuleHandleW(name) }
    }

    /// # Safety
    ///
    /// `name` is null or points to a NUL-terminated string valid for reads.
    pub(super) unsafe fn get_module_handle_a(name: *const u8) -> *mut c_void {
        // SAFETY: the pointers are as this function's contract gives them.
        unsafe { GetModuleHandleA(name) }
    }

    /// # Safety
    ///
    /// `out` is valid for writes of `size` characters.
    pub(super) unsafe fn get_module_file_name_w(
        module: *mut c_void,
        out: *mut u16,
        size: u32,
    ) -> u32 {
        // SAFETY: the pointers are as this function's contract gives them.
        unsafe { GetModuleFileNameW(module, out, size) }
    }

    /// # Safety
    ///
    /// `out` is valid for writes of `size` bytes.
    pub(super) unsafe fn get_module_file_name_a(
        module: *mut c_void,
        out: *mut u8,
        size: u32,
    ) -> u32 {
        // SAFETY: the pointers are as this function's contract gives them.
        unsafe { GetModuleFileNameA(module, out, size) }
    }

    pub(super) fn set_last_error(code: u32) {
        // SAFETY: see above.
        unsafe { SetLastError(code) }
    }

    /// # Safety
    ///
    /// `name` is a NUL-terminated string valid for reads, and `file` is
    /// null.
    pub(super) unsafe fn load_library_ex_a(
        name: *const u8,
        file: *mut c_void,
        flags: u32,
    ) -> *mut c_void {
        // SAFETY: the pointers are as this function's contract gives them.
        unsafe { LoadLibraryExA(name, file, flags) }
    }

    /// # Safety
    ///
    /// `name` is a NUL-terminated UTF-16 string valid for reads, and `file`
    /// is null.
    pub(super) unsafe fn load_library_ex_w(
        name: *const u16,
        file: *mut c_void,
        flags: u32,
    ) -> *mut c_void {
        // SAFETY: the pointers are as this function's contract gives them.
        unsafe { LoadLibraryExW(name, file, flags) }
    }

    /// # Safety
    ///
    /// `name` is an ordinal or a NUL-terminated string valid for reads.
    pub(super) unsafe fn get_proc_address(module: *mut c_void, name: *const u8) -> *mut c_void {
        // SAFETY: the pointers are as this function's contract gives them.
        unsafe { GetProcAddress(module, name) }.map_or(std::ptr::null_mut(), |f| f as *mut c_void)
    }

    pub(super) fn free_library(module: *mut c_void) -> i32 {
        // SAFETY: see above.
        unsafe { FreeLibrary(module) }
    }

    pub(super) fn disable_thread_library_calls(module: *mut c_void) -> i32 {
        match victory_loader::disable_thread_calls(module as usize) {
            Some(done) => i32::from(done),
            // SAFETY: see above.
            None => unsafe { DisableThreadLibraryCalls(module) },
        }
    }

    pub(super) fn exit(code: i32) -> ! {
        // SAFETY: see above.
        unsafe { crt_exit(code) }
    }

    pub(super) fn abort() -> ! {
        // SAFETY: see above.
        unsafe { crt_abort() }
    }

    pub(super) fn terminate_process(process: *mut c_void, code: u32) -> i32 {
        // SAFETY: see above.
        unsafe { TerminateProcess(process, code) }
    }

    pub(super) fn set_unhandled_exception_filter(filter: extern "system" fn(*const c_void) -> i32) {
        // SAFETY: our filter takes the exception pointers as an opaque pointer.
        unsafe {
            windows_sys::Win32::System::Diagnostics::Debug::SetUnhandledExceptionFilter(
                std::mem::transmute::<
                    extern "system" fn(*const c_void) -> i32,
                    windows_sys::Win32::System::Diagnostics::Debug::LPTOP_LEVEL_EXCEPTION_FILTER,
                >(filter),
            )
        };
    }

    /// # Safety
    ///
    /// `info` is the exception information the system passed to our filter.
    pub(super) unsafe fn report_unhandled(info: *const c_void) {
        // SAFETY: the system's exception pointers, per this function's
        // contract.
        unsafe { victory_loader::report_unhandled(info.cast()) };
    }

    pub(super) fn note_title_output(line: &str) {
        victory_loader::note_title_output(line);
    }

    /// Our caller's callers, as `module+rva`.
    pub(super) fn callers() -> String {
        let mut frames = [std::ptr::null_mut(); 12];
        // SAFETY: fills at most `frames.len()` return addresses.
        let n = unsafe {
            RtlCaptureStackBackTrace(
                2,
                frames.len() as u32,
                frames.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        let names: Vec<_> =
            frames[..usize::from(n)].iter().map(|&f| victory_loader::locate(f as usize)).collect();
        names.join(" < ")
    }
}

#[cfg(not(windows))]
mod host {
    use std::ffi::c_void;

    /// # Safety
    ///
    /// `name` is null or points to a NUL-terminated UTF-16 string valid for
    /// reads.
    pub(super) unsafe fn get_module_handle_w(_: *const u16) -> *mut c_void {
        std::ptr::null_mut()
    }

    /// # Safety
    ///
    /// `name` is null or points to a NUL-terminated string valid for reads.
    pub(super) unsafe fn get_module_handle_a(_: *const u8) -> *mut c_void {
        std::ptr::null_mut()
    }

    /// # Safety
    ///
    /// `name` is a NUL-terminated string valid for reads, and `file` is
    /// null.
    pub(super) unsafe fn load_library_ex_a(_: *const u8, _: *mut c_void, _: u32) -> *mut c_void {
        std::ptr::null_mut()
    }

    /// # Safety
    ///
    /// `out` is valid for writes of `size` characters.
    pub(super) unsafe fn get_module_file_name_w(_: *mut c_void, _: *mut u16, _: u32) -> u32 {
        0
    }

    /// # Safety
    ///
    /// `out` is valid for writes of `size` bytes.
    pub(super) unsafe fn get_module_file_name_a(_: *mut c_void, _: *mut u8, _: u32) -> u32 {
        0
    }

    pub(super) fn set_last_error(_: u32) {}

    /// # Safety
    ///
    /// `name` is a NUL-terminated UTF-16 string valid for reads, and `file`
    /// is null.
    pub(super) unsafe fn load_library_ex_w(_: *const u16, _: *mut c_void, _: u32) -> *mut c_void {
        std::ptr::null_mut()
    }

    /// # Safety
    ///
    /// `name` is an ordinal or a NUL-terminated string valid for reads.
    pub(super) unsafe fn get_proc_address(_: *mut c_void, _: *const u8) -> *mut c_void {
        std::ptr::null_mut()
    }

    pub(super) fn free_library(_: *mut c_void) -> i32 {
        0
    }

    pub(super) fn disable_thread_library_calls(_: *mut c_void) -> i32 {
        0
    }

    #[allow(clippy::exit, reason = "the stand-in for the CRT's `exit`, which the title asked for")]
    pub(super) fn exit(code: i32) -> ! {
        std::process::exit(code)
    }

    pub(super) fn abort() -> ! {
        std::process::abort()
    }

    #[allow(
        clippy::exit,
        reason = "the stand-in for `TerminateProcess`, which the title asked for"
    )]
    pub(super) fn terminate_process(_: *mut c_void, code: u32) -> i32 {
        std::process::exit(code as i32)
    }

    pub(super) fn set_unhandled_exception_filter(_: extern "system" fn(*const c_void) -> i32) {}

    /// # Safety
    ///
    /// `info` is the exception information the system passed to our filter.
    pub(super) unsafe fn report_unhandled(_: *const c_void) {}

    pub(super) fn note_title_output(_: &str) {}

    pub(super) fn callers() -> String {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_images_report_their_original_path() {
        let base = 0x7000_0000_0000;
        runtime::set_image_path(base, std::path::Path::new(r"C:\title\game.exe"));
        let module = base as *mut c_void;
        let mut buf = [0xFFFFu16; 32];
        // SAFETY: `buf` is a local buffer of `buf.len()` characters.
        let n = unsafe { GetModuleFileNameW(module, buf.as_mut_ptr(), buf.len() as u32) };
        assert_eq!(String::from_utf16_lossy(&buf[..n as usize]), r"C:\title\game.exe");
        assert_eq!(buf[n as usize], 0);

        // Too small: truncated, NUL-terminated, the buffer size returned.
        let mut small = [0u8; 4];
        // SAFETY: `small` is a local buffer of 4 bytes.
        assert_eq!(unsafe { GetModuleFileNameA(module, small.as_mut_ptr(), 4) }, 4);
        assert_eq!(&small, &[b'C', b':', b'\\', 0]);
    }

    fn w(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(Some(0)).collect()
    }

    // The exports, over arguments that are valid by construction.
    fn handle_w(text: &str) -> *mut c_void {
        // SAFETY: `w` ends the string with a NUL, and it outlives the call.
        unsafe { GetModuleHandleW(w(text).as_ptr()) }
    }
    fn handle_a(text: &CStr) -> *mut c_void {
        // SAFETY: a `CStr` is NUL-terminated.
        unsafe { GetModuleHandleA(text.as_ptr().cast()) }
    }
    fn load_w(text: &str) -> *mut c_void {
        // SAFETY: as `handle_w`.
        unsafe { LoadLibraryW(w(text).as_ptr()) }
    }
    fn load_a(text: &CStr) -> *mut c_void {
        // SAFETY: as `handle_a`.
        unsafe { LoadLibraryA(text.as_ptr().cast()) }
    }
    fn load_ex_w(text: &str, flags: u32) -> *mut c_void {
        // SAFETY: as `handle_w`; the reserved argument is null.
        unsafe { LoadLibraryExW(w(text).as_ptr(), std::ptr::null_mut(), flags) }
    }
    #[cfg(windows)]
    fn proc_address(module: *mut c_void, name: &CStr) -> *mut c_void {
        // SAFETY: a `CStr` is NUL-terminated.
        unsafe { GetProcAddress(module, name.as_ptr().cast()) }
    }

    #[cfg(windows)]
    #[test]
    fn host_lookups_by_name_get_our_overrides() {
        let kernel32 = handle_w("kernel32.dll");
        assert!(!kernel32.is_null());
        let ours = crate::registry::lookup("kernel32.dll", "CreateFileW").unwrap();
        assert_eq!(proc_address(kernel32, c"CreateFileW") as usize, ours);
        // A plain name stays the host's.
        // SAFETY: a NUL-terminated literal.
        let host = unsafe { host::get_proc_address(kernel32, c"GetTickCount".as_ptr().cast()) };
        assert!(!host.is_null());
        assert_eq!(proc_address(kernel32, c"GetTickCount"), host);
        // Host DLLs outside the host policy do not exist for the title.
        assert!(load_w("wininet.dll").is_null());
        assert!(load_a(c"C:\\Windows\\System32\\urlmon").is_null());
        assert!(handle_w("WININET.DLL").is_null());
        assert_eq!(
            // SAFETY: reads this thread's last error.
            unsafe { windows_sys::Win32::Foundation::GetLastError() },
            ERROR_MOD_NOT_FOUND
        );
        assert!(!load_w("kernel32").is_null());
        // By ordinal too (ws2_32 #4 is `connect`).
        let ws2 = load_w("ws2_32.dll");
        // SAFETY: an ordinal, which is never dereferenced.
        let by_ordinal = unsafe { GetProcAddress(ws2, 4 as *const u8) };
        assert_eq!(by_ordinal as usize, crate::registry::lookup("ws2_32.dll", "connect").unwrap());
    }

    #[test]
    fn title_dlls_and_pseudo_modules_are_found_by_name() {
        let base = 0x7100_0000_0000usize;
        runtime::set_image_path(base, std::path::Path::new(r"C:\title\Mapped.dll"));
        runtime::set_title_dll_loader(Box::new(|file| match file {
            "late.dll" => Ok(Some(0x7200_0000_0000)),
            "broken.dll" => Err("bad image".into()),
            _ => Ok(None),
        }));
        let mapped = base as *mut c_void;
        assert_eq!(handle_w("MAPPED"), mapped);
        assert_eq!(handle_a(c"sub\\mapped.dll"), mapped);
        assert_eq!(load_w("mapped.dll"), mapped);
        assert_eq!(FreeLibrary(mapped), 1, "title images stay loaded");

        // Not mapped yet: `LoadLibrary` maps it, `GetModuleHandle` does not.
        assert!(handle_w("late.dll").is_null());
        assert_eq!(load_a(c"late") as usize, 0x7200_0000_0000);
        assert!(load_a(c"broken.dll").is_null());
        // As data it runs no code, so the host loads it (and finds nothing).
        let data = load_ex_w("late.dll", 0x2);
        assert_ne!(data as usize, 0x7200_0000_0000);

        let runtime = load_a(c"xgameruntime.dll");
        assert!(!runtime.is_null());
        assert_eq!(handle_w("XGameRuntime.dll"), runtime);
    }
}

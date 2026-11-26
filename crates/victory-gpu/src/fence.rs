//! Fences: a 64-bit value and the events waiting for it. Nothing runs on a
//! GPU yet, so only signals (the title's, later our queues') move one.

use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::object::{self, E_NOTIMPL, FENCE, HRESULT, Iid, Object, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

pub(crate) const FENCE_IIDS: [Iid; 4] = [
    guid(0x0a753dcf, 0xc4d8, 0x4b91, [0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76]),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Fence {
    object: Object,
    /// Titles also read it straight from memory, at the fence's GPU
    /// address.
    value: AtomicU64,
    flags: u32,
    /// Events to set once the value reaches theirs (handles as integers).
    waiters: Mutex<Vec<(u64, usize)>>,
}

/// # Safety
///
/// `this` is a live fence this module created, and it stays alive for `'a`
/// (fences are never destroyed while the title holds a reference).
unsafe fn fence<'a>(this: This) -> &'a Fence {
    // SAFETY: `this` is one of our live fences, per this function's contract.
    unsafe { &*this.cast::<Fence>() }
}

/// Device method.
///
/// # Safety
///
/// `iid` is null or points to an interface id valid for reads, and `out` is
/// null or valid for a write of an object pointer, for the duration of the
/// call.
pub(crate) unsafe extern "system" fn create(
    _: This,
    initial: u64,
    flags: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::debug!(initial, flags, "fence");
    let fence = Box::new(Fence {
        object: Object::new(&FUNCTIONS),
        value: AtomicU64::new(initial),
        flags,
        waiters: Mutex::new(Vec::new()),
    });
    // SAFETY: the pointers passed on are under this function's own contract.
    unsafe { object::hand_out(FENCE, fence, iid, &FENCE_IIDS, out) }
}

fn set_event(event: usize) {
    #[cfg(windows)]
    // SAFETY: the title passed an event handle.
    unsafe {
        windows_sys::Win32::System::Threading::SetEvent(event as *mut c_void)
    };
    #[cfg(not(windows))]
    let _ = event;
}

/// `QueryInterface`.
///
/// # Safety
///
/// `this` is a live fence this module created. `iid` is null or points to
/// an interface id valid for reads, and `out` is null or valid for a write
/// of an object pointer, for the duration of the call.
unsafe extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    // SAFETY: the pointers passed on are under this function's own contract.
    unsafe { object::answer(FENCE, iid, object::is(iid, &FENCE_IIDS).then_some(this), out) }
}
/// `Release`.
///
/// # Safety
///
/// `this` is a live fence this module created, holding the reference the
/// caller gives up; it must not be used after the last one is released.
unsafe extern "system" fn release(this: This) -> u32 {
    // SAFETY: a live object this module made with `hand_out` from a box of
    // this type (this function's contract).
    unsafe { object::release_boxed::<Fence>(this) }
}
/// `GetCompletedValue`.
///
/// # Safety
///
/// `this` is a live fence this module created.
unsafe extern "system" fn completed(this: This) -> u64 {
    // SAFETY: `this` is a live fence, per this function's contract.
    unsafe { fence(this) }.value.load(Ordering::Acquire)
}
/// `SetEventOnCompletion`.
///
/// # Safety
///
/// `this` is a live fence this module created. `event` is null or an event
/// handle valid until the fence reaches `value`.
unsafe extern "system" fn set_event_on_completion(
    this: This,
    value: u64,
    event: *mut c_void,
) -> HRESULT {
    // SAFETY: `this` is a live fence, per this function's contract.
    let fence = unsafe { fence(this) };
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    if fence.value.load(Ordering::Acquire) >= value {
        if !event.is_null() {
            set_event(event as usize);
        }
        return S_OK;
    }
    if event.is_null() {
        // shortcut: a null event means "block until reached"; nothing can
        // reach it while no GPU work runs, so fail instead of hanging.
        tracing::warn!(value, "blocking fence wait not implemented");
        return E_NOTIMPL;
    }
    waiters.push((value, event as usize));
    S_OK
}
/// `Signal`.
///
/// # Safety
///
/// `this` is a live fence this module created.
pub(crate) unsafe extern "system" fn signal(this: This, value: u64) -> HRESULT {
    // SAFETY: `this` is a live fence, per this function's contract.
    let fence = unsafe { fence(this) };
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    fence.value.store(value, Ordering::Release);
    waiters.retain(|&(wanted, event)| {
        let reached = wanted <= value;
        if reached {
            set_event(event);
        }
        !reached
    });
    S_OK
}

/// # Safety
///
/// `this` is a live fence this module created.
unsafe extern "system" fn creation_flags(this: This) -> u32 {
    // SAFETY: `this` is a live fence, per this function's contract.
    unsafe { fence(this) }.flags
}
/// Where the value lives (GPU addresses are CPU addresses).
///
/// # Safety
///
/// `this` is a live fence this module created.
unsafe extern "system" fn gpu_address(this: This) -> u64 {
    // SAFETY: `this` is a live fence, per this function's contract.
    let fence = unsafe { fence(this) };
    (&raw const fence.value) as u64
}

/// IUnknown 3, object 4, device child 1, fence 3, fence 1 (2, a console
/// one among them).
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<FENCE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = completed as *const (); // 0x40
    f[9] = set_event_on_completion as *const (); // 0x48
    f[10] = signal as *const (); // 0x50
    f[11] = creation_flags as *const (); // 0x58
    f[12] = gpu_address as *const (); // 0x60
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    #[cfg(windows)]
    #[test]
    fn signals_set_waiting_events() {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let mut f = std::ptr::null();
        // SAFETY: a static interface id and a local to write.
        assert_eq!(unsafe { create(std::ptr::null(), 1, 0, &FENCE_IIDS[0], &mut f) }, S_OK);
        let completed = slot::<unsafe extern "system" fn(This) -> u64>(f, 0x40);
        // SAFETY: `f` is the fence just created; every call below uses it
        // before its last reference is released at the end.
        assert_eq!(unsafe { completed(f) }, 1);
        // SAFETY: plain auto-reset event.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        assert!(!event.is_null());
        let wait = slot::<unsafe extern "system" fn(This, u64, *mut c_void) -> HRESULT>(f, 0x48);
        let signal = slot::<unsafe extern "system" fn(This, u64) -> HRESULT>(f, 0x50);
        // SAFETY: `f` is live and `event` is a valid event handle.
        assert_eq!(unsafe { wait(f, 5, event) }, S_OK);
        // SAFETY: a valid event.
        assert_ne!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        // SAFETY: `f` is live.
        assert_eq!(unsafe { signal(f, 5) }, S_OK);
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        // SAFETY: as above.
        assert_eq!(unsafe { wait(f, 5, event) }, S_OK, "already reached");
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        // SAFETY: as above; null is a tolerated event.
        assert_eq!(unsafe { wait(f, 9, std::ptr::null_mut()) }, E_NOTIMPL);
        // Grand Theft Auto VI reads the value at the fence's address.
        let gpu_address = slot::<unsafe extern "system" fn(This) -> u64>(f, 0x60);
        // SAFETY: `f` is live.
        let at = unsafe { gpu_address(f) };
        // SAFETY: the fence's value.
        assert_eq!(unsafe { *(at as *const u64) }, 5);
        // SAFETY: ours.
        unsafe { CloseHandle(event) };
        let release = slot::<unsafe extern "system" fn(This) -> u32>(f, 0x10);
        // SAFETY: `f` is live; this releases the only reference.
        assert_eq!(unsafe { release(f) }, 0);
    }
}

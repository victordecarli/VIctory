//! Memory the frame report names: the commit charge of the process and of
//! the system, and the parts of it this crate holds. A run aborts when the
//! system's commit reaches its limit, so the report shows how close it is
//! and how much of the process is ours, and how much the title's XMem
//! memory.
//!
//! shortcut: what a list holds until it is reset (upload buffers, scratch,
//! `allocate_upload` blocks) is not counted; it lives one submission.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// What this crate keeps in system memory for as long as the title's
/// objects live.
#[derive(Clone, Copy)]
pub(crate) enum Held {
    /// Host copies of title buffers and of title memory outside buffers
    /// (`crate::shadow`).
    Copies,
    /// Bytes of host copies kept before a submission to tell what its
    /// shaders wrote, in use or pooled (`crate::shadow`).
    Snapshots,
    /// Texture staging buffers and the reads of title memory detiled into
    /// them (`crate::texture`).
    Staging,
}

static HELD: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

impl Held {
    fn counter(self) -> &'static AtomicU64 {
        let i = match self {
            Held::Copies => 0,
            Held::Snapshots => 1,
            Held::Staging => 2,
        };
        &HELD[i]
    }

    /// `bytes` more are held.
    pub(crate) fn hold(self, bytes: u64) {
        self.counter().fetch_add(bytes, Ordering::Relaxed);
    }

    /// `bytes` held before are given back.
    pub(crate) fn release(self, bytes: u64) {
        self.counter().fetch_sub(bytes, Ordering::Relaxed);
    }

    pub(crate) fn bytes(self) -> u64 {
        self.counter().load(Ordering::Relaxed)
    }
}

/// The title's reservations (`XMemVirtualAlloc`), by start: their length.
static RESERVED: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());

/// The title reserved `[base, base + len)` (XMem virtual memory): what it
/// commits there is counted as the title's.
pub fn title_reserved(base: usize, len: usize) {
    RESERVED.lock().unwrap_or_else(|e| e.into_inner()).insert(base, len);
}

/// Bytes committed in `[base, base + len)`: private, then mapped (views of
/// sections, which the system's commit counts and the process's does not).
/// `None` once nothing is reserved at `base`.
fn committed_in(base: usize, len: usize) -> Option<[u64; 2]> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_FREE, MEM_MAPPED, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    let end = base.saturating_add(len);
    let (mut at, mut sums) = (base, [0u64; 2]);
    while at < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size; any address may be
        // asked about.
        if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0 {
            break;
        }
        if at == base && info.State == MEM_FREE {
            return None;
        }
        let next = (info.BaseAddress as usize + info.RegionSize.max(1)).min(end);
        if info.State == MEM_COMMIT {
            let bytes = (next - at) as u64;
            if info.Type == MEM_PRIVATE {
                sums[0] += bytes;
            } else if info.Type == MEM_MAPPED {
                sums[1] += bytes;
            }
        }
        at = next;
    }
    Some(sums)
}

/// What the title's reservations commit now, private and mapped; released
/// ones are forgotten.
///
/// shortcut: memory the title takes outside XMem (the host's
/// `VirtualAlloc`, which it reaches directly) is not counted, and a page
/// mapped at two addresses counts twice.
fn title_memory() -> [u64; 2] {
    let mut reserved = RESERVED.lock().unwrap_or_else(|e| e.into_inner());
    let mut total = [0u64; 2];
    reserved.retain(|&base, &mut len| match committed_in(base, len) {
        Some([private, mapped]) => {
            total[0] += private;
            total[1] += mapped;
            true
        }
        None => false,
    });
    total
}

/// Memory now, in bytes.
#[derive(Clone, Copy)]
pub(crate) struct Memory {
    /// The process's commit charge (`PagefileUsage`): the title's memory,
    /// ours and the driver's.
    pub(crate) commit: u64,
    /// The system's commit charge and its limit (RAM plus paging files).
    pub(crate) system_commit: u64,
    pub(crate) commit_limit: u64,
    pub(crate) copies: u64,
    pub(crate) snapshots: u64,
    pub(crate) staging: u64,
    /// Committed in the title's XMem reservations: private (part of
    /// `commit`), then views of its physical pages (not part of it).
    pub(crate) title_private: u64,
    pub(crate) title_mapped: u64,
}

impl Memory {
    /// The counts now; a figure Windows does not give is 0.
    pub(crate) fn now() -> Memory {
        use windows_sys::Win32::System::ProcessStatus::{
            K32GetPerformanceInfo, K32GetProcessMemoryInfo, PERFORMANCE_INFORMATION,
            PROCESS_MEMORY_COUNTERS,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let mut process = PROCESS_MEMORY_COUNTERS {
            cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
            ..Default::default()
        };
        // SAFETY: our own process and a sized structure.
        let commit = if unsafe {
            K32GetProcessMemoryInfo(GetCurrentProcess(), &mut process, process.cb)
        } != 0
        {
            process.PagefileUsage as u64
        } else {
            tracing::debug!(error = %std::io::Error::last_os_error(), "no process memory counters");
            0
        };
        let mut system = PERFORMANCE_INFORMATION {
            cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
            ..Default::default()
        };
        // SAFETY: a sized structure.
        let (system_commit, commit_limit) = if unsafe {
            K32GetPerformanceInfo(&mut system, system.cb)
        } != 0
        {
            let page = system.PageSize as u64;
            (system.CommitTotal as u64 * page, system.CommitLimit as u64 * page)
        } else {
            tracing::debug!(error = %std::io::Error::last_os_error(), "no system memory counters");
            (0, 0)
        };
        let [title_private, title_mapped] = title_memory();
        Memory {
            commit,
            system_commit,
            commit_limit,
            copies: Held::Copies.bytes(),
            snapshots: Held::Snapshots.bytes(),
            staging: Held::Staging.bytes(),
            title_private,
            title_mapped,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Held, Memory};

    #[test]
    fn what_is_held_is_given_back() {
        // Snapshots: no other test of this crate takes one.
        let before = Held::Snapshots.bytes();
        Held::Snapshots.hold(3 << 20);
        assert_eq!(Held::Snapshots.bytes(), before + (3 << 20));
        Held::Snapshots.release(3 << 20);
        assert_eq!(Held::Snapshots.bytes(), before);
    }

    #[test]
    fn title_memory_counts_what_its_reservations_commit_private_and_mapped() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE,
            MapViewOfFile, PAGE_READWRITE, SEC_RESERVE, UnmapViewOfFile, VirtualAlloc, VirtualFree,
        };
        const MIB: usize = 1 << 20;
        // SAFETY: a fresh reservation of our own, then a part of it
        // committed.
        let private = unsafe {
            let at = VirtualAlloc(std::ptr::null(), 4 * MIB, MEM_RESERVE, PAGE_READWRITE);
            assert!(!at.is_null());
            assert!(!VirtualAlloc(at, MIB, MEM_COMMIT, PAGE_READWRITE).is_null());
            at as usize
        };
        // SAFETY: a reserved section of our own, a view of all of it, and
        // its first 64 KiB committed.
        let (section, mapped) = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                MIB as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, MIB);
            assert!(!view.Value.is_null());
            assert!(!VirtualAlloc(view.Value, 64 << 10, MEM_COMMIT, PAGE_READWRITE).is_null());
            (section, view)
        };
        let view = mapped.Value as usize;
        super::title_reserved(private, 4 * MIB);
        super::title_reserved(view, MIB);
        assert_eq!(super::committed_in(private, 4 * MIB), Some([MIB as u64, 0]));
        assert_eq!(super::committed_in(view, MIB), Some([0, 64 << 10]));
        // SAFETY: what this test made, given back.
        unsafe {
            assert!(VirtualFree(private as *mut _, 0, MEM_RELEASE) != 0);
            assert!(UnmapViewOfFile(mapped) != 0);
            CloseHandle(section);
        }
        // Released reservations are no longer the title's.
        assert_eq!(super::committed_in(private, 4 * MIB), None);
        super::title_memory();
        let reserved = super::RESERVED.lock().unwrap();
        assert!(!reserved.contains_key(&private) && !reserved.contains_key(&view));
    }

    #[test]
    fn the_process_commit_is_part_of_the_system_commit_within_its_limit() {
        let Memory {
            commit,
            system_commit,
            commit_limit,
            copies: _,
            snapshots: _,
            staging: _,
            title_private: _,
            title_mapped: _,
        } = Memory::now();
        assert!(commit > 0, "the process commits memory");
        assert!(commit <= system_commit, "{commit} within {system_commit}");
        assert!(system_commit <= commit_limit, "{system_commit} within {commit_limit}");
    }
}

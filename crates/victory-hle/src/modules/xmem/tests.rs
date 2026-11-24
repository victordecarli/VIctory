use super::*;

fn query(address: *const c_void) -> BasicInformation {
    // SAFETY: plain data; zero is a valid value.
    let mut info: BasicInformation = unsafe { std::mem::zeroed() };
    // SAFETY: a pseudo handle needs no closing.
    let process = unsafe { GetCurrentProcess() };
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { XMemVirtualQuery(process, address, &mut info) }, 1);
    info
}

#[test]
fn alloc_honours_alignment_and_frees() {
    for (field, align) in [(0, 16), (4, 16), (0xc, 4096), (0x10, 64 * 1024)] {
        let attributes = field << ALIGNMENT_SHIFT | 1 << 29; // a graphics type
        let p = XMemAlloc(100, attributes);
        assert!(!p.is_null());
        assert_eq!(p as usize % align, 0, "field {field:#x}");
        // SAFETY: 100 bytes were allocated.
        unsafe { p.cast::<u8>().write_bytes(0xab, 100) };
        // SAFETY: the arguments are this test's locals, valid for the call.
        unsafe { XMemFree(p, attributes) };
    }
    let p = XMemAlloc(4096, ALLOC_32BIT);
    assert!(!p.is_null() && (p as usize) < 1 << 32);
    // SAFETY: the arguments are this test's locals, valid for the call.
    unsafe { XMemFree(p, ALLOC_32BIT) };
}

#[test]
fn virtual_alloc_meets_console_constraints() {
    const GPU_READ_WRITE: u32 = 0x2000;
    // SAFETY: the arguments are this test's locals, valid for the call.
    let p = unsafe {
        XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * MIB,
            PAGES_2M | MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | 7 << 12, // 40-bit addresses
            PAGE_READWRITE | GPU_READ_WRITE,
        )
    };
    assert!(!p.is_null());
    assert_eq!(p as usize % (2 * MIB), 0);
    assert!((p as usize) < 1 << 40);
    let info = query(p);
    assert_eq!(info.host.State, MEM_COMMIT);
    assert_eq!(info.host.Protect, PAGE_READWRITE);
    assert_eq!(info.flags & GRAPHICS, GRAPHICS);
    // Write-watched, for the GPU side's copies.
    let (mut found, mut count, mut granularity) = ([0usize; 4], 4, 0);
    // SAFETY: room for `count` addresses.
    let watched = unsafe {
        windows_sys::Win32::System::Memory::GetWriteWatch(
            0,
            p,
            4 * MIB,
            found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    };
    assert_eq!(watched, 0);
    // SAFETY: our own reservation.
    assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
    assert_eq!(query(p).flags, 0);
    // No address field: above the 32-bit range and below 2^40, where
    // the console keeps it.
    // SAFETY: the arguments are this test's locals, valid for the call.
    let p = unsafe {
        XMemVirtualAlloc(std::ptr::null_mut(), 16 << 30, MEM_RESERVE, GRAPHICS, PAGE_NOACCESS)
    };
    assert!(!p.is_null() && (p as usize) >= 1 << 32, "{p:p}");
    assert!((p as usize) + (16 << 30) <= 1 << 40, "{p:p}");
    // SAFETY: our own reservation.
    assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
}

#[test]
fn transfers_shrink_the_title_budget() {
    let mut before = WorkingSet::default();
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { XMemGetWorkingSetStatistics(0x2000_0001, &mut before) }, 1);
    assert!(before.title_used > 0);
    assert_eq!(XMemTransferMemory(0, 128 * MIB - 1), 1);
    assert_eq!(XMemTransferMemory(2, MIB), 0, "no tools memory");
    let standard = victory_gpu::console::current().title_memory().1;
    assert_eq!(XMemTransferMemory(0, standard), 0, "over budget");
    let mut after = WorkingSet::default();
    // SAFETY: the arguments are this test's locals, valid for the call.
    unsafe { XMemGetWorkingSetStatistics(0x2000_0001, &mut after) };
    assert_eq!(before.title_limit - after.title_limit, 128 * MIB);
}

#[test]
fn statistics_have_the_console_layout() {
    assert_eq!(size_of::<WorkingSet>(), 0x50);
    assert_eq!(size_of::<BasicInformation>(), 0x40);
    assert_eq!(std::mem::offset_of!(WorkingSet, gpu_optimal_limit), 0x28);
}

#[test]
fn commit_limit_report_includes_request_and_system_commit() {
    let error = std::io::Error::from_raw_os_error(ERROR_COMMITMENT_LIMIT as i32);
    let info = PERFORMANCE_INFORMATION {
        CommitTotal: 512,
        CommitLimit: 768,
        PageSize: 4096,
        ..Default::default()
    };
    let report = commit_limit_message(4 * MIB, &error, Some(&info));
    assert!(report.contains("4194304 bytes"));
    assert!(report.contains("2 MiB of 3 MiB in use"));
    let fallback = commit_limit_message(4 * MIB, &error, None);
    for report in [&report, &fallback] {
        assert!(report.contains("os error 1455"));
        assert!(report.contains("Windows paging file size"));
    }
    assert!(!fallback.contains("System commit:"));
}

#[test]
fn physical_pages_map_alias_and_unmap() {
    let mut pages = [0usize; 3];
    let mut n = 3;
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()) }, 1);
    assert_eq!(n, 3);
    // Packed into 21 bits as `number >> 4`, as the title keeps them.
    for p in pages {
        assert_eq!((p >> 4 & 0x1f_ffff) << 4, p);
    }
    // Reservations of 64 KiB pages, as titles make them for mapping.
    let reserve = || {
        // SAFETY: the arguments are this test's locals, valid for the call.
        unsafe {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        }
    };
    let (a, b) = (reserve(), reserve());
    assert!(!a.is_null() && !b.is_null());
    let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { XMemMapPhysicalPages(a, 3, pages.as_ptr()) }, a);

    // The same page at a second address shows the same bytes.
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { XMemMapPhysicalPages(at(b, 1), 1, pages[2..].as_ptr()) }, at(b, 1));
    // SAFETY: both pages are mapped read-write.
    unsafe {
        at(a, 2).cast::<u8>().write(7);
        assert_eq!(at(b, 1).cast::<u8>().read(), 7);
        a.cast::<u8>().write(1);
    }
    // Writes into pages (DirectStorage's) show through their views.
    assert!(write_physical(&pages[2..], 1, &[9]));
    // SAFETY: mapped read-write.
    assert_eq!(unsafe { at(b, 1).cast::<u8>().add(1).read() }, 9);

    // Unmapping the middle page keeps its neighbours.
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { XMemMapPhysicalPages(at(a, 1), 1, std::ptr::null()) }, at(a, 1));
    assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
    assert_eq!(query(a).host.State, MEM_COMMIT);
    assert_eq!(query(at(a, 2)).host.State, MEM_COMMIT);
    // SAFETY: still mapped.
    assert_eq!(unsafe { at(a, 2).cast::<u8>().read() }, 7);

    // A scattered page lands in the hole.
    let addresses = [at(a, 1)];
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, pages.as_ptr()) }, 1);
    // SAFETY: page 0 is now mapped at both `a` and `a + PAGE`.
    assert_eq!(unsafe { at(a, 1).cast::<u8>().read() }, 1);
    // Page number 0 unmaps.
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, [0usize].as_ptr()) }, 1);
    assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
    // The GPU's page-mapping copy: what `a + 2 pages` shows, at `b`.
    assert!(map_alias(b as u64, at(a, 2) as u64, PAGE as u64));
    // SAFETY: mapped read-write.
    assert_eq!(unsafe { b.cast::<u8>().read() }, 7);
    assert!(map_alias(b as u64, 0, PAGE as u64));
    assert_eq!(query(b).host.State, MEM_RESERVE);
    assert!(!map_alias(b as u64, at(a, 1) as u64, PAGE as u64));
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { XMemFreePhysicalPages(3, pages.as_ptr()) }, 1);
    assert_eq!(XMemIsToolsMemAvailable(), 0);
}

#[test]
fn mapping_no_pages_changes_nothing() {
    let pages = allocate(2);
    // SAFETY: the arguments are this test's locals, valid for the call.
    let a = unsafe {
        XMemVirtualAlloc(std::ptr::null_mut(), 2 * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE)
    };
    assert!(!a.is_null());
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { XMemMapPhysicalPages(a, 2, pages.as_ptr()) }, a);
    // SAFETY: both pages are mapped read-write.
    unsafe { a.cast::<u8>().write(5) };
    let second = a.wrapping_byte_add(PAGE);

    // A count of 0 at the start of a view, inside it and past it: with
    // pages, without (an unmap), and as the GPU's page-mapping copy.
    for at in [a, second, a.wrapping_byte_add(2 * PAGE)] {
        // SAFETY: the arguments are this test's locals, valid for the call.
        assert_eq!(unsafe { XMemMapPhysicalPages(at, 0, pages.as_ptr()) }, at);
        // SAFETY: the arguments are this test's locals, valid for the call.
        assert_eq!(unsafe { XMemMapPhysicalPages(at, 0, std::ptr::null()) }, at);
        assert!(map_alias(at as u64, a as u64, 0));
        assert!(map_alias(at as u64, 0, 0));
    }

    // The view is still the one view it was.
    let view = physical().views.get(&(a as usize)).map(|v| v.0);
    assert_eq!(view, Some(2 * PAGE));
    assert_eq!(query(a).host.State, MEM_COMMIT);
    assert_eq!(query(second).host.State, MEM_COMMIT);
    // SAFETY: still mapped.
    assert_eq!(unsafe { a.cast::<u8>().read() }, 5);

    // A count of 1 still unmaps that page alone.
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { XMemMapPhysicalPages(second, 1, std::ptr::null()) }, second);
    assert_eq!(query(second).host.State, MEM_RESERVE);
    assert_eq!(query(a).host.State, MEM_COMMIT);
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { XMemMapPhysicalPages(a, 1, std::ptr::null()) }, a);
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { XMemFreePhysicalPages(2, pages.as_ptr()) }, 1);
}

/// `n` pages of one new allocation.
fn allocate(n: usize) -> Vec<usize> {
    let (mut pages, mut given) = (vec![0; n], n);
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()) }, 1);
    assert_eq!(given, n);
    pages
}

/// The bytes of one page, through a view of our own.
fn page_bytes(page: usize) -> Vec<u8> {
    let slice = physical().pages.get(&page).cloned().expect("known page");
    View::map(&slice, 1).expect("mapped").bytes().to_vec()
}

#[test]
fn runs_group_consecutive_pages_of_one_allocation() {
    let (a, b) = (allocate(3), allocate(1));
    let lengths = |pages: &[usize], offset, len| {
        runs(pages, offset, len).map(|r| r.iter().map(|&(_, n)| n).collect::<Vec<_>>())
    };
    assert_eq!(lengths(&a, 0, 3 * PAGE), Some(vec![3]));
    // Only the pages the bytes reach.
    assert_eq!(lengths(&a, PAGE + 5, 10), Some(vec![1]));
    assert_eq!(lengths(&a, PAGE - 1, 2), Some(vec![2]));
    assert_eq!(lengths(&a, 0, 0), Some(vec![]));
    // Out of order, or across allocations: one run each.
    assert_eq!(lengths(&[a[0], a[2], a[1]], 0, 3 * PAGE), Some(vec![1, 1, 1]));
    assert_eq!(lengths(&[a[1], a[2], b[0]], 0, 3 * PAGE), Some(vec![2, 1]));
    // A slice that follows on in another section starts a run.
    assert_eq!(lengths(&[b[0], a[1]], 0, 2 * PAGE), Some(vec![1, 1]));
    // Too few pages for the bytes, or an unknown one.
    assert_eq!(lengths(&a, 0, 3 * PAGE + 1), None);
    assert_eq!(lengths(&[a[0], 0x10], 0, 2 * PAGE), None);
    assert_eq!(lengths(&a, usize::MAX, 2), None);
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { XMemFreePhysicalPages(3, a.as_ptr()) }, 1);
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { XMemFreePhysicalPages(1, b.as_ptr()) }, 1);
}

#[test]
fn writes_cross_runs_and_allocations() {
    let (a, b) = (allocate(2), allocate(1));
    let pages = [a[0], a[1], b[0]];
    let data: Vec<u8> = (0..2 * PAGE + 64).map(|i| (i * 7 % 251) as u8).collect();
    let offset = PAGE - 32;
    // Past the third page: the second run is too short.
    assert!(!write_physical(&pages, offset, &data));
    assert!(write_physical(&pages, offset, &data[..2 * PAGE]));
    let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
    assert!(seen[..offset].iter().all(|&b| b == 0), "before the offset");
    assert_eq!(seen[offset..offset + 2 * PAGE], data[..2 * PAGE]);
    assert!(seen[offset + 2 * PAGE..].iter().all(|&b| b == 0), "after");
    assert!(!write_physical(&[a[0], 0x10], 0, &data[..PAGE + 1]), "unknown page");
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { XMemFreePhysicalPages(2, a.as_ptr()) }, 1);
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { XMemFreePhysicalPages(1, b.as_ptr()) }, 1);
}

#[test]
fn page_windows_cover_one_run_in_place() {
    let (a, b) = (allocate(3), allocate(1));
    let mut window = page_window(&a, PAGE + 3, PAGE).expect("one run");
    assert_eq!(window.bytes().len(), PAGE);
    window.bytes().fill(0x5a);
    drop(window);
    let (second, third) = (page_bytes(a[1]), page_bytes(a[2]));
    assert!(second[..3].iter().all(|&b| b == 0));
    assert!(second[3..].iter().all(|&b| b == 0x5a));
    assert!(third[..3].iter().all(|&b| b == 0x5a));
    assert!(third[3..].iter().all(|&b| b == 0));
    // Several runs, or an unknown page: no window.
    assert!(page_window(&[a[0], b[0]], 0, PAGE + 1).is_none());
    assert!(page_window(&[a[1], a[0]], 0, PAGE + 1).is_none());
    assert!(page_window(&[0x10], 0, 1).is_none());
    // One page of several is still one run.
    assert!(page_window(&[a[0], b[0]], 0, PAGE).is_some());
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { XMemFreePhysicalPages(3, a.as_ptr()) }, 1);
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { XMemFreePhysicalPages(1, b.as_ptr()) }, 1);
}

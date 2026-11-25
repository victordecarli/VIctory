use std::os::windows::ffi::OsStrExt;
use std::time::{Duration, Instant};

use super::*;
use crate::modules::xgameruntime::method;

fn call<F: Copy>(object: *mut c_void, slot: usize) -> F {
    let object = crate::test_support::valid_mut(object);
    // SAFETY: every object here starts with the runtime header.
    method(unsafe { &*object.cast::<Object>() }, slot)
}

fn factory() -> *mut c_void {
    let mut factory = std::ptr::null_mut();
    // SAFETY: this test's own locals, valid for the call.
    assert_eq!(unsafe { DStorageGetFactory(&FACTORY, &mut factory) }, S_OK);
    crate::test_support::valid_mut(factory)
}

fn open(path: &std::path::Path) -> *mut c_void {
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
    let open: extern "system" fn(*mut c_void, *const u16, *const Guid, Out) -> HRESULT =
        call(factory(), 0x20);
    let mut file = std::ptr::null_mut();
    assert_eq!(open(factory(), wide.as_ptr(), std::ptr::null(), &mut file), S_OK);
    file
}

fn status_array(capacity: u32) -> *mut c_void {
    let create: extern "system" fn(*mut c_void, u32, *const c_char, *const Guid, Out) -> HRESULT =
        call(factory(), 0x28);
    let mut array = std::ptr::null_mut();
    let name = c"test".as_ptr();
    assert_eq!(create(factory(), capacity, name, std::ptr::null(), &mut array), S_OK);
    crate::test_support::valid_mut(array)
}

fn new_queue(source_type: u64) -> *mut c_void {
    let create: extern "system" fn(*mut c_void, *const QueueDesc, *const Guid, Out) -> HRESULT =
        call(factory(), 0x50);
    let desc =
        QueueDesc { source_type, capacity: 0x80, priority: 0, name: c"test".as_ptr(), unknown: 1 };
    let mut queue = std::ptr::null_mut();
    assert_eq!(create(factory(), &desc, std::ptr::null(), &mut queue), S_OK);
    crate::test_support::valid_mut(queue)
}

fn request(
    options: u64,
    destination: *mut c_void,
    destination_size: usize,
    source: *const c_void,
    offset: u64,
    source_size: usize,
) -> Request {
    Request {
        options,
        destination,
        page_offset: 0,
        destination_size: destination_size as u32,
        source,
        offset,
        source_size: source_size as u32,
        intermediate_size: 0,
        tag: 0,
        name: std::ptr::null(),
    }
}

fn enqueue(queue: *mut c_void, request: &Request) {
    let f: extern "system" fn(*mut c_void, *const Request) = call(queue, 0x18);
    f(queue, request);
}

fn enqueue_status(queue: *mut c_void, array: *mut c_void, index: u32) {
    let f: extern "system" fn(*mut c_void, *mut c_void, u32) = call(queue, 0x20);
    f(queue, array, index);
}

fn void(object: *mut c_void, slot: usize) {
    let f: extern "system" fn(*mut c_void) = call(object, slot);
    f(object);
}

fn result(array: *mut c_void, index: u32) -> HRESULT {
    let f: extern "system" fn(*mut c_void, u32) -> HRESULT = call(array, 0x20);
    f(array, index)
}

/// Polls as the title does, then returns the entry's result.
fn wait(array: *mut c_void, index: u32) -> HRESULT {
    let done: extern "system" fn(*mut c_void, u32) -> bool = call(array, 0x18);
    let start = Instant::now();
    while !done(array, index) {
        assert!(start.elapsed() < Duration::from_secs(10), "never completed");
        std::thread::sleep(Duration::from_millis(1));
    }
    result(array, index)
}

fn file_with(bytes: &[u8]) -> (tempfile::TempDir, *mut c_void) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.bin");
    std::fs::write(&path, bytes).unwrap();
    let file = open(&path);
    (dir, file)
}

fn pattern(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i * 7 + i / 251) as u8).collect()
}

#[test]
fn request_layout_is_the_titles() {
    assert_eq!(size_of::<Request>(), 0x48);
    assert_eq!(std::mem::offset_of!(Request, destination_size), 0x18);
    assert_eq!(std::mem::offset_of!(Request, source), 0x20);
    assert_eq!(std::mem::offset_of!(Request, source_size), 0x30);
    assert_eq!(std::mem::offset_of!(Request, name), 0x40);
    assert_eq!(std::mem::offset_of!(QueueDesc, unknown), 0x18);
}

#[test]
fn raw_and_zlib_reads_from_a_file() {
    let plain = pattern(300_000);
    let packed = miniz_oxide::deflate::compress_to_vec_zlib(&plain, 6);
    let mut contents = plain.clone();
    contents.extend_from_slice(&packed);
    let (_dir, file) = file_with(&contents);
    let (queue, array) = (new_queue(0), status_array(1));

    let mut raw = vec![0u8; 1000];
    let mut inflated = vec![0u8; plain.len()];
    let at = |b: &mut Vec<u8>| b.as_mut_ptr().cast();
    enqueue(queue, &request(0, at(&mut raw), 1000, file, 5000, 1000));
    let offset = plain.len() as u64;
    let zlib = request(ZLIB, at(&mut inflated), plain.len(), file, offset, packed.len());
    enqueue(queue, &zlib);
    enqueue_status(queue, array, 0);
    assert_eq!(result(array, 0), E_PENDING, "pending until done");
    void(queue, 0x30);
    assert_eq!(wait(array, 0), S_OK);
    assert_eq!(raw, plain[5000..6000]);
    assert_eq!(inflated, plain);
    void(queue, 0x40);
}

#[test]
fn memory_sources_inflate_too() {
    let plain = pattern(70_000);
    let packed = miniz_oxide::deflate::compress_to_vec_zlib(&plain, 6);
    let (queue, array) = (new_queue(1), status_array(1));
    let mut out = vec![0u8; plain.len()];
    let options = ZLIB | MEMORY_SOURCE;
    let r = request(
        options,
        out.as_mut_ptr().cast(),
        out.len(),
        packed.as_ptr().cast(),
        0,
        packed.len(),
    );
    enqueue(queue, &r);
    enqueue_status(queue, array, 0);
    void(queue, 0x30);
    assert_eq!(wait(array, 0), S_OK);
    assert_eq!(out, plain);
}

#[test]
fn statuses_complete_in_order_with_the_first_failure() {
    let contents = pattern(4096);
    let (_dir, file) = file_with(&contents);
    let (queue, array) = (new_queue(0), status_array(3));
    // Requests in flight together run in any order, so none of them
    // shares a destination (failed ones leave theirs undefined).
    let mut buffers = [[0u8; 16]; 5];
    let [failed, failed_too, out, other, bcpack] =
        buffers.each_mut().map(|b| b.as_mut_ptr().cast());

    // Segment 0: a read past the end, then a bad stream; the first wins.
    enqueue(queue, &request(0, failed, 16, file, 4090, 16));
    enqueue(queue, &request(ZLIB, failed_too, 16, file, 0, 16));
    enqueue_status(queue, array, 0);
    // Segment 1: two reads.
    enqueue(queue, &request(0, other, 16, file, 0, 16));
    enqueue(queue, &request(0, out, 16, file, 100, 16));
    enqueue_status(queue, array, 1);
    void(queue, 0x30);
    // Segment 2: a BCPack request (BC7) whose stream is empty fails.
    with_test_tables();
    enqueue(queue, &request(7 << 1, bcpack, 16, file, 0, 0));
    enqueue_status(queue, array, 2);
    void(queue, 0x30);

    assert_eq!(wait(array, 2), INVALID_DATA);
    assert_eq!(result(array, 0), END_OF_FILE);
    assert_eq!(result(array, 1), S_OK);
    assert_eq!(buffers[2], contents[100..116]);
    assert_eq!(buffers[3], contents[..16]);
}

#[test]
fn half_a_queue_submits_itself() {
    let (_dir, file) = file_with(&pattern(64));
    let create: extern "system" fn(*mut c_void, *const QueueDesc, *const Guid, Out) -> HRESULT =
        call(factory(), 0x50);
    let desc =
        QueueDesc { source_type: 0, capacity: 4, priority: 1, name: c"small".as_ptr(), unknown: 1 };
    let mut queue = std::ptr::null_mut();
    assert_eq!(create(factory(), &desc, std::ptr::null(), &mut queue), S_OK);
    let array = status_array(1);
    let mut out = [0u8; 16];
    enqueue(queue, &request(0, out.as_mut_ptr().cast(), 16, file, 8, 16));
    enqueue_status(queue, array, 0); // the second of four: no submit needed
    assert_eq!(wait(array, 0), S_OK);
    assert_eq!(out, pattern(64)[8..24]);
}

/// Lanes without workers: what `start` picks, and when statuses land.
fn lanes(staging: u64) -> Lanes {
    Lanes {
        lanes: HashMap::new(),
        next_id: 0,
        next_seq: 0,
        staged: 0,
        staging,
        starved: false,
        clock: [0; 3],
        now: 0,
        lane_now: [0; 3],
        done: Default::default(),
        refused: 0,
        bcpack: Default::default(),
        since: None,
    }
}

fn read(lanes: &mut Lanes, id: u64, options: u64, size: u32) {
    let mut r = request(options, std::ptr::null_mut(), 0, std::ptr::null(), 0, 0);
    (r.destination_size, r.source_size) = (size, size);
    let lane = lanes.lanes.get_mut(&id).unwrap();
    lane.unsubmitted.push(Command::Read(r));
}

#[test]
fn priorities_share_by_bytes_about_tenfold() {
    let mut l = lanes(STAGING);
    l.lanes.insert(0, Lane::new(0, 4096));
    l.lanes.insert(1, Lane::new(1, 4096));
    l.lanes.insert(2, Lane::new(2, 4096)); // real-time: as high
    for _ in 0..400 {
        for id in 0..3 {
            read(&mut l, id, 0, 4096);
        }
    }
    (0..3).for_each(|id| l.submit(id));
    let mut started = [0u32; 3];
    for _ in 0..330 {
        let job = l.start().unwrap();
        started[job.lane as usize] += 1;
        l.finish(&job, Ok(()));
    }
    // Normal gets about a tenth of what high and real-time get together.
    assert_eq!(started[1] + started[2], 300, "{started:?}");
    assert_eq!(started[0], 30, "{started:?}");
    assert!(started[1].abs_diff(started[2]) <= 1, "{started:?}");
}

#[test]
fn staging_bounds_compressed_bytes_in_flight() {
    let mut l = lanes(100);
    l.lanes.insert(0, Lane::new(0, 64));
    read(&mut l, 0, ZLIB, 60);
    read(&mut l, 0, ZLIB, 60);
    read(&mut l, 0, 0, 1000); // raw: straight to its destination
    l.submit(0);
    let first = l.start().unwrap();
    assert!(l.start().is_none(), "60 + 60 > 100");
    l.finish(&first, Ok(()));
    let second = l.start().unwrap();
    let raw = l.start().unwrap();
    assert_eq!((second.staging, raw.staging), (60, 0));
    // Alone, a request larger than the buffer still runs.
    read(&mut l, 0, ZLIB, 500);
    l.submit(0);
    assert!(l.start().is_none());
    l.finish(&second, Ok(()));
    assert!(l.start().is_some());
}

#[test]
fn a_status_waits_for_every_earlier_read() {
    let mut l = lanes(STAGING);
    l.lanes.insert(0, Lane::new(0, 64));
    let array = status_array(2).cast::<Instance<StatusArray>>();
    let status = |index| {
        // SAFETY: this test's own locals, valid for the call.
        unsafe { add_ref_instance(array) };
        Command::Status(Status { array, index })
    };
    read(&mut l, 0, 0, 16);
    read(&mut l, 0, 0, 16);
    l.lanes.get_mut(&0).unwrap().unsubmitted.push(status(0));
    read(&mut l, 0, 0, 16);
    l.lanes.get_mut(&0).unwrap().unsubmitted.push(status(1));
    // SAFETY: an array we created.
    for entry in unsafe { &(*array).state.entries }.iter() {
        entry.store(E_PENDING, Ordering::Release);
    }
    l.submit(0);
    let jobs: Vec<Job> = (0..3).map(|_| l.start().unwrap()).collect();
    // The last read and the first finish; the second then fails.
    l.finish(&jobs[2], Ok(()));
    l.finish(&jobs[0], Ok(()));
    let results = || (result(array.cast(), 0), result(array.cast(), 1));
    assert_eq!(results(), (E_PENDING, E_PENDING));
    l.finish(&jobs[1], Err(INVALID_DATA));
    assert_eq!(results(), (INVALID_DATA, S_OK));
}

#[test]
fn zlib_then_bcpack() {
    with_test_tables();
    let packed = miniz_oxide::deflate::compress_to_vec_zlib(&BC4_STREAM, 6);
    let (_dir, file) = file_with(&packed);
    let (queue, array) = (new_queue(0), status_array(1));
    let mut out = [0u8; 16];
    let options = ZLIB | 4 << 1;
    let mut r = request(options, out.as_mut_ptr().cast(), 16, file, 0, packed.len());
    r.intermediate_size = BC4_STREAM.len() as u32;
    enqueue(queue, &r);
    enqueue_status(queue, array, 0);
    void(queue, 0x30);
    assert_eq!(wait(array, 0), S_OK);
    assert_eq!(out, bc4_blocks());
}

/// BCPack tables we made (uniform models; BC4 index pairs: low 3 bits,
/// next 3 bits), in a folder kept for the test process.
fn with_test_tables() {
    static DIR: OnceLock<()> = OnceLock::new();
    DIR.get_or_init(|| {
        let mut t = Vec::new();
        for m in 0..128 {
            let g = m / 16;
            for b in 0..9 {
                let f: u16 = if b < g + 2 { 1 << (15 - g) } else { 0 };
                t.extend_from_slice(&f.to_le_bytes());
            }
        }
        t.extend([0; 256 + 64 + 256]);
        t.extend((0..64u8).map(|v| (v & 7) | (v >> 3) << 4));
        t.resize(bcpack::TABLES_SIZE, 0);
        let dir = tempfile::tempdir().unwrap().keep();
        std::fs::write(dir.join("tables.bin"), t).unwrap();
        use_bcpack_tables(dir);
    });
}

/// Two BC4 blocks from the format's test encoder: endpoints 10, -3 with
/// indices p -> p & 7, then the same block plus 1 everywhere.
const BC4_STREAM: [u8; 88] = [
    0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 4, 4, 4, 4, 1, 1, 1, 1, 0x10, 0x10, 0x10, 0x10, 2, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0x14, 5, 0x20, 0x68, 0xb0, 0xf8, 3, 3, 2, 2, 0x24, 0x24, 0x24, 0x24, 2, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 0x10, 0x10, 0x10, 0x10, 0, 0, 0, 0, 0x20, 0x68, 0xb0,
    0xf8, 0x24, 0x24, 0x24, 0x24, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// What `BC4_STREAM` unpacks to.
fn bc4_blocks() -> [u8; 16] {
    let code = [0u64, 2, 3, 4, 5, 6, 7, 1];
    let block = |e0: u64, e1: u64, add: usize| {
        let bits = (0..16).fold(0, |a, p| a | code[(p + add) & 7] << (16 + 3 * p));
        (e0 | e1 << 8 | bits).to_le_bytes()
    };
    let mut out = [0; 16];
    out[..8].copy_from_slice(&block(138, 125, 0));
    out[8..].copy_from_slice(&block(139, 126, 1));
    out
}

#[test]
fn bcpack_requests_unpack_into_the_destination() {
    with_test_tables();
    let mut contents = pattern(100);
    contents.extend_from_slice(&BC4_STREAM);
    let (_dir, file) = file_with(&contents);
    let (queue, array) = (new_queue(0), status_array(1));
    let mut out = [0u8; 16];
    enqueue(queue, &request(4 << 1, out.as_mut_ptr().cast(), 16, file, 100, 88));
    enqueue_status(queue, array, 0);
    void(queue, 0x30);
    assert_eq!(wait(array, 0), S_OK);
    assert_eq!(out, bc4_blocks());
}

#[test]
fn a_closed_queue_completes_nothing() {
    let (_dir, file) = file_with(&pattern(64));
    let (queue, array) = (new_queue(0), status_array(1));
    let mut out = [0u8; 16];
    enqueue(queue, &request(0, out.as_mut_ptr().cast(), 16, file, 0, 16));
    enqueue_status(queue, array, 0);
    void(queue, 0x40);
    void(queue, 0x30);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(result(array, 0), E_PENDING);
    assert_eq!(out, [0; 16]);
}

#[test]
fn physical_page_destinations() {
    use crate::modules::xmem;
    let mut pages = [0usize; 2];
    let mut n = 2;
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { xmem::XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()) }, 1);
    // Where the title would see them: a reservation of 64 KiB pages.
    // SAFETY: the arguments are this test's locals, valid for the call.
    let mapped = unsafe {
        xmem::XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            0x2000 | 0x2040_0000, // reserve, 64 KiB pages
            0,
            0x04,
        )
    };
    assert!(!mapped.is_null());
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { xmem::XMemMapPhysicalPages(mapped, 2, pages.as_ptr()) }, mapped);

    let plain = pattern(100);
    let (queue, array) = (new_queue(1), status_array(1));
    let mut r = request(
        PAGE_DESTINATION | MEMORY_SOURCE,
        pages.as_mut_ptr().cast(),
        plain.len(),
        plain.as_ptr().cast(),
        0,
        plain.len(),
    );
    r.page_offset = (PAGE - 40) as u64; // across the page boundary
    enqueue(queue, &r);
    enqueue_status(queue, array, 0);
    void(queue, 0x30);
    assert_eq!(wait(array, 0), S_OK);
    // SAFETY: both pages are mapped read-write.
    let seen = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>().add(PAGE - 40), 100) };
    assert_eq!(seen, plain);
}
#[test]
fn zlib_into_pages_in_place_and_across_allocations() {
    use crate::modules::xmem;
    let allocate = |n: usize| {
        let (mut pages, mut given) = (vec![0usize; n], n);
        let (given_ptr, pages_ptr) = (&mut given, pages.as_mut_ptr());
        // SAFETY: the arguments are this test's locals, valid for the call.
        assert_eq!(unsafe { xmem::XMemAllocatePhysicalPages(0, given_ptr, pages_ptr) }, 1);
        pages
    };
    let (a, b) = (allocate(3), allocate(1));
    let plain = pattern(2 * PAGE + 100);
    let packed = miniz_oxide::deflate::compress_to_vec_zlib(&plain, 6);
    // One allocation's pages in order (one window), then pages of two
    // (through a buffer), both from 50 bytes into the first.
    let mut destinations = [a.clone(), vec![a[0], a[1], b[0]]];
    let (queue, array) = (new_queue(1), status_array(1));
    for pages in &mut destinations {
        let mut r = request(
            ZLIB | PAGE_DESTINATION | MEMORY_SOURCE,
            pages.as_mut_ptr().cast(),
            plain.len(),
            packed.as_ptr().cast(),
            0,
            packed.len(),
        );
        r.page_offset = 50;
        enqueue(queue, &r);
    }
    enqueue_status(queue, array, 0);
    void(queue, 0x30);
    assert_eq!(wait(array, 0), S_OK);
    for pages in &destinations {
        // SAFETY: the arguments are this test's locals, valid for the call.
        let mapped = unsafe {
            xmem::XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                0x2000 | 0x2040_0000, // reserve, 64 KiB pages
                0,
                0x04,
            )
        };
        assert!(!mapped.is_null());
        // SAFETY: the arguments are this test's locals, valid for the call.
        assert_eq!(unsafe { xmem::XMemMapPhysicalPages(mapped, 3, pages.as_ptr()) }, mapped);
        // SAFETY: all three pages are mapped read-write.
        let seen = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>(), 3 * PAGE) };
        assert!(seen[..50].iter().all(|&b| b == 0));
        assert_eq!(seen[50..50 + plain.len()], plain);
        // SAFETY: the arguments are this test's locals, valid for the call.
        assert_eq!(unsafe { xmem::XMemMapPhysicalPages(mapped, 3, std::ptr::null()) }, mapped);
    }
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { xmem::XMemFreePhysicalPages(3, a.as_ptr()) }, 1);
    // SAFETY: the arguments are this test's locals, valid for the call.
    assert_eq!(unsafe { xmem::XMemFreePhysicalPages(1, b.as_ptr()) }, 1);
}

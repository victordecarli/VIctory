use super::*;
use crate::object::slot;

type Release = extern "system" fn(This) -> u32;
type Map = extern "system" fn(This, u32, *const c_void, *mut *mut u8) -> HRESULT;
type Address = extern "system" fn(This) -> u64;

const PROPERTIES: HeapProperties =
    HeapProperties { kind: 1, cpu_pages: 0, pool: 0, creation_nodes: 0, visible_nodes: 0 };

fn buffer(width: u64) -> ResourceDesc {
    ResourceDesc {
        dimension: BUFFER,
        alignment: 0,
        width,
        height: 1,
        depth_or_array_size: 1,
        mip_levels: 1,
        format: 0,
        sample_count: 1,
        sample_quality: 0,
        layout: 1,
        flags: 0,
    }
}

fn texture(width: u64, height: u32, format: u32, mips: u16) -> ResourceDesc {
    ResourceDesc { dimension: 3, width, height, format, mip_levels: mips, layout: 0, ..buffer(0) }
}

fn heap(size: u64) -> This {
    let desc = HeapDesc { size, properties: PROPERTIES, alignment: 0, flags: 0 };
    let mut heap = std::ptr::null();
    // SAFETY: this test's own locals and resources, live for the call.
    assert_eq!(unsafe { create_heap(std::ptr::null(), &desc, &HEAP_IIDS[0], &mut heap) }, S_OK);
    crate::test_support::valid(heap)
}

#[test]
fn heaps_are_zeroed_aligned_memory() {
    let heap = heap(1 << 20);
    // SAFETY: a heap we just made.
    let memory = unsafe { &(*heap.cast::<Heap>()).memory };
    assert_eq!(memory.ptr.as_ptr() as u64 % KIB_64, 0);
    let mut back = std::mem::MaybeUninit::<HeapDesc>::uninit();
    slot::<extern "system" fn(This, *mut HeapDesc)>(heap, 0x40)(heap, back.as_mut_ptr());
    // SAFETY: written above.
    assert_eq!(unsafe { back.assume_init() }.size, 1 << 20);
    assert_eq!(slot::<Release>(heap, 0x10)(heap), 0);

    let bad = HeapDesc { size: 1 << 20, properties: PROPERTIES, alignment: 3, flags: 0 };
    let mut out = std::ptr::null();
    assert_eq!(
        // SAFETY: a local description, a static id and a local to write.
        unsafe { create_heap(std::ptr::null(), &bad, &HEAP_IIDS[0], &mut out) },
        E_INVALIDARG
    );
}

#[test]
fn device_setup_heap_properties_and_resources_at_addresses() {
    let device = crate::device::device();
    // GetCustomHeapProperties (0xd0): the struct comes back through the
    // hidden first argument, which is also returned.
    type Custom = extern "system" fn(*mut HeapProperties, This, u32, u32) -> *mut HeapProperties;
    let mut p = PROPERTIES;
    let back = slot::<Custom>(device, 0xd0)(&mut p, device, 1, 2);
    assert_eq!(back, &raw mut p);
    assert_eq!(
        (p.kind, p.cpu_pages, p.pool, p.creation_nodes, p.visible_nodes),
        (4, 3, 1, 1, 1),
        "upload: custom, write-back, L0"
    );
    slot::<Custom>(device, 0xd0)(&mut p, device, 1, 1);
    assert_eq!(p.cpu_pages, 1, "default: not CPU-visible");

    // 0x428: a buffer over memory the title owns, as GTA VI asks.
    type At = extern "system" fn(
        This,
        u64,
        *const ResourceDesc,
        u32,
        *const c_void,
        *const Iid,
        *mut This,
    ) -> HRESULT;
    let memory = vec![0u8; 0x1_0000];
    let address = memory.as_ptr() as u64;
    let mut r = std::ptr::null();
    let create = slot::<At>(device, 0x428);
    let desc = buffer(0x1_0000);
    let null = std::ptr::null();
    assert_eq!(create(device, address, &desc, u32::MAX, null, &RESOURCE_IIDS[0], &mut r), 0);
    assert_eq!(slot::<Address>(r, 0x58)(r), address);
    assert_eq!(slot::<Release>(r, 0x10)(r), 0);
    // The title's memory is still there after the release.
    assert_eq!(memory[0xffff], 0);
    assert_ne!(create(device, 0, &desc, u32::MAX, null, &RESOURCE_IIDS[0], &mut r), 0);
}

#[test]
fn sizes_cover_every_mip() {
    let size = |d: &ResourceDesc| footprint(d).map(|i| i.size);
    assert_eq!(size(&buffer(256)), Some(KIB_64));
    // Textures take their console layout and its alignment (4 KiB
    // standard swizzle when the description leaves it to the library):
    // 512x512 BC7 with 10 mips is 0x56000 bytes, 4 KiB aligned, as on
    // the console.
    let layout = |d: &ResourceDesc| texture_layout(d).unwrap().size;
    let rgba = texture(256, 256, 28, 0);
    assert_eq!(size(&rgba), Some(layout(&rgba).next_multiple_of(4096)));
    let bc7 = footprint(&texture(512, 512, 98, 10));
    let console = AllocationInfo { size: 0x56000, alignment: 0x1000 };
    assert_eq!(bc7, Some(console));
    assert!(layout(&rgba) >= 4 * (256 * 256 * 4) / 3, "every mip");
    // BC1 is 8 bytes per 4x4 block: 1024x1024, one mip = 512 KiB.
    assert_eq!(size(&texture(1024, 1024, 71, 1)), Some(512 << 10));
    assert_eq!(size(&texture(16, 16, 66, 1)), None, "1-bit format");
    // Linear layout (1) and an explicit mode (64 KiB display, 0x10a).
    let linear = ResourceDesc { layout: 1, ..texture(100, 60, 28, 1) };
    assert_eq!(texture_layout(&linear).unwrap().planes[0].layout.desc.swizzle.0, 0);
    let display = ResourceDesc { layout: 0x10a, ..texture(100, 60, 28, 1) };
    assert_eq!(texture_layout(&display).unwrap().planes[0].layout.desc.swizzle.0, 10);

    let mut info = AllocationInfo { size: 0, alignment: 0 };
    let descs = [buffer(100), texture(256, 256, 28, 0)];
    // SAFETY: this test's own locals and resources, live for the call.
    unsafe { allocation_info(&mut info, std::ptr::null(), 0, 2, descs.as_ptr()) };
    assert_eq!(info, AllocationInfo { size: KIB_64 + size(&descs[1]).unwrap(), alignment: KIB_64 });
    // SAFETY: this test's own locals and resources, live for the call.
    unsafe { allocation_info(&mut info, std::ptr::null(), 0, 1, &texture(1, 1, 999, 1)) };
    assert_eq!(info.size, u64::MAX);
}

#[test]
fn upload_footprints_follow_the_pc_rules() {
    let null = std::ptr::null();
    let mut layouts = [NO_FOOTPRINT; 2];
    let (mut rows, mut sizes, mut total) = ([0; 2], [0; 2], 0);
    let mut footprints = |desc: &ResourceDesc, first, count| {
        let (l, r, s) = (layouts.as_mut_ptr(), rows.as_mut_ptr(), sizes.as_mut_ptr());
        // SAFETY: this test's own locals and resources, live for the call.
        unsafe { copyable_footprints(null, desc, first, count, 0, l, r, s, &mut total) };
        (layouts, rows, sizes, total)
    };
    // 100x60 RGBA8, 2 mips: 400-byte rows pitched to 512.
    let (l, r, s, t) = footprints(&texture(100, 60, 28, 2), 0, 2);
    assert_eq!((l[0].row_pitch, r[0], s[0]), (512, 60, 400));
    // Mip 1 (50x30, 200-byte rows) starts after 59 pitched rows and one
    // packed row, rounded up to 512.
    assert_eq!(l[1].offset, (512 * 59 + 400u64).next_multiple_of(512));
    assert_eq!((l[1].width, l[1].height, r[1]), (50, 30, 30));
    assert_eq!(t, l[1].offset + 256 * 29 + 200);
    // BC1: rows of 4x4 blocks, 8 bytes each.
    let (_, r, s, _) = footprints(&texture(64, 64, 71, 1), 0, 1);
    assert_eq!((r[0], s[0]), (16, 128));
    let (_, _, _, t) = footprints(&texture(64, 64, 71, 1), 1, 1);
    assert_eq!(t, u64::MAX, "no second subresource");
}

/// The two 2560x1440 targets the title creates at its "press Enter"
/// screen, as the run log shows them after `pc_format`, each with the
/// clear value a render target comes with.
#[cfg(windows)]
#[test]
fn full_screen_targets_get_a_host_texture() {
    if crate::host::host().is_none() {
        return; // no host GPU here
    }
    const RENDER_TARGET: u32 = 1;
    const UNORDERED: u32 = 4;
    // (format, flags asked, flags the host copy has): B5G5R5A1_UNORM has
    // no typed UAV on PC, B4G4R4A4_UNORM no render target or UAV; a format
    // the host supports in full keeps every flag.
    let cases = [
        (86, RENDER_TARGET | UNORDERED, RENDER_TARGET),
        (115, RENDER_TARGET, 0),
        (28, RENDER_TARGET | UNORDERED, RENDER_TARGET | UNORDERED),
    ];
    let wrong: Vec<_> = cases
        .into_iter()
        .filter_map(|(format, flags, kept)| {
            let desc = ResourceDesc { flags, ..texture(2560, 1440, format, 1) };
            let clear: [u32; 5] = [format, 0, 0, 0, 0];
            // SAFETY: a whole clear value.
            let made = unsafe { host_texture(&desc, 0, clear.as_ptr().cast()) };
            // SAFETY: a live resource.
            let has = made.map(|t| unsafe { t.GetDesc() }.Flags.0 as u32);
            (has != Some(kept)).then_some((format, flags, has))
        })
        .collect();
    assert!(wrong.is_empty(), "(format, flags asked, host flags) {wrong:?}");
}

/// A 4x4 texture filled from a title buffer through our list and queue,
/// then read back from the host GPU.
#[cfg(windows)]
#[test]
fn buffer_data_reaches_the_host_texture() {
    use windows::Win32::Graphics::Direct3D12::*;
    let Some(host) = crate::host::host() else {
        return; // no host GPU here
    };
    let null = std::ptr::null();
    let iid = &RESOURCE_IIDS[0];
    const COPY_DEST: u32 = 0x400;
    let (mut image, mut source) = (null, null);
    let rgba = texture(4, 4, 28, 1);
    assert_eq!(
        // SAFETY: this test's own locals and resources, live for the call.
        unsafe {
            create_committed(null, &PROPERTIES, 0, &rgba, COPY_DEST, null.cast(), iid, &mut image)
        },
        S_OK
    );
    assert_eq!(
        // SAFETY: this test's own locals and resources, live for the call.
        unsafe {
            create_committed(null, &PROPERTIES, 0, &buffer(1024), 0, null.cast(), iid, &mut source)
        },
        S_OK
    );
    // SAFETY: this test's own locals and resources, live for the call.
    let (data, _) = unsafe { buffer_bytes(source) }.unwrap();
    // SAFETY: the buffer holds 1024 bytes.
    let bytes = unsafe { std::slice::from_raw_parts_mut(data, 1024) };
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }

    let queue_iid =
        guid(0x0ec870a6, 0x5d7e, 0x4c22, [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed]);
    let allocator_iid =
        guid(0x6102dee4, 0xaf59, 0x4b09, [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24]);
    let list_iid =
        guid(0x5b160d0f, 0xac1b, 0x4185, [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55]);
    let (mut queue, mut allocator, mut list) = (null, null, null);
    let queue_desc = [0u32; 4];
    // SAFETY: a local description of 4 words, a local id and output.
    let created =
        unsafe { crate::queue::create(null, queue_desc.as_ptr().cast(), &queue_iid, &mut queue) };
    assert_eq!(created, S_OK);
    assert_eq!(
        // SAFETY: this test's own locals, valid for the call.
        unsafe { crate::list::create_allocator(null, 0, &allocator_iid, &mut allocator) },
        S_OK
    );
    assert_eq!(
        // SAFETY: this test's own locals, valid for the call.
        unsafe { crate::list::create_list(null, 0, 0, allocator, null, &list_iid, &mut list) },
        S_OK
    );

    #[repr(C)]
    struct Location {
        resource: This,
        kind: u32,
        footprint: PlacedFootprint,
    }
    let rows =
        PlacedFootprint { offset: 0, format: 28, width: 4, height: 4, depth: 1, row_pitch: 256 };
    let to =
        Location { resource: image, kind: 0, footprint: PlacedFootprint { offset: 0, ..rows } };
    let from = Location { resource: source, kind: 1, footprint: rows };
    type Copy =
        extern "system" fn(This, *const Location, u32, u32, u32, *const Location, *const u8);
    slot::<Copy>(list, 0x80)(list, &to, 0, 0, 0, &from, null.cast());
    slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list);
    slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);

    // Read the host texture back into a readback buffer.
    let readback_heap =
        D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let readback_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 1024,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut readback: Option<ID3D12Resource> = None;
    // SAFETY: plain creation of a readback buffer.
    unsafe {
        host.device
            .CreateCommittedResource(
                &readback_heap,
                D3D12_HEAP_FLAG_NONE,
                &readback_desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut readback,
            )
            .unwrap();
    }
    let readback = readback.unwrap();
    // SAFETY: this test's own locals and resources, live for the call.
    let host_image = unsafe { host_resource(image) }.unwrap();
    let back = host.new_list().unwrap();
    // SAFETY: this test's own locals and resources, live for the call.
    let (_, before) = unsafe { host_transition(image, 0x800) }.unwrap();
    // The upload list holds the texture until it is reset.
    // SAFETY: this test's own locals and resources, live for the call.
    assert!(std::sync::Arc::strong_count(&unsafe { host_copy(image) }.unwrap()) > 2);
    use windows::core::Interface;
    let out_to =
        Location { resource: readback.as_raw().cast_const().cast(), kind: 1, footprint: rows };
    let out_from = Location {
        resource: host_image.as_raw().cast_const().cast(),
        kind: 0,
        footprint: to.footprint,
    };
    // SAFETY: an open list, live resources, locations in the host layout.
    unsafe {
        back.list.ResourceBarrier(&[crate::host::transition(
            host_image,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        back.list.CopyTextureRegion(
            (&raw const out_to).cast(),
            0,
            0,
            0,
            (&raw const out_from).cast(),
            None,
        );
    }
    back.close();
    host.execute(&[&back]);
    let mut mapped = std::ptr::null_mut();
    // SAFETY: the copy finished (execute waits); the buffer is 1024 bytes.
    let read = unsafe {
        readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 1024 }), Some(&mut mapped)).unwrap();
        std::slice::from_raw_parts(mapped.cast::<u8>(), 1024)
    };
    for row in 0..4 {
        let at = row * 256;
        assert_eq!(read[at..at + 16], bytes[at..at + 16], "row {row}");
    }
}

#[test]
fn committed_and_placed_resources_map_at_their_gpu_address() {
    let null = std::ptr::null();
    let mut committed = null;
    // SAFETY: this test's own locals and resources, live for the call.
    let hr = unsafe {
        create_committed(
            null,
            &PROPERTIES,
            0,
            &buffer(256),
            0,
            null.cast(),
            &RESOURCE_IIDS[0],
            &mut committed,
        )
    };
    assert_eq!(hr, S_OK);
    let mut data = std::ptr::null_mut();
    assert_eq!(slot::<Map>(committed, 0x40)(committed, 0, null.cast(), &mut data), S_OK);
    assert_eq!(slot::<Address>(committed, 0x58)(committed), data as u64);
    // SAFETY: the buffer is at least 256 zeroed bytes.
    assert_eq!(unsafe { *data.add(255) }, 0);
    assert_eq!(slot::<Release>(committed, 0x10)(committed), 0);

    let heap = heap(4 * KIB_64);
    let place = |offset, desc: &ResourceDesc, out: &mut This| {
        // SAFETY: this test's own locals and resources, live for the call.
        unsafe { create_placed(null, heap, offset, desc, 0, null.cast(), &RESOURCE_IIDS[0], out) }
    };
    let mut placed = null;
    assert_eq!(place(KIB_64, &buffer(KIB_64), &mut placed), S_OK);
    // SAFETY: a heap we made.
    let base = unsafe { (*heap.cast::<Heap>()).memory.ptr.as_ptr() } as u64;
    assert_eq!(slot::<Address>(placed, 0x58)(placed), base + KIB_64);
    let mut other = null;
    assert_eq!(place(4 * KIB_64, &buffer(1), &mut other), E_INVALIDARG, "past the end");
    assert_eq!(place(100, &buffer(1), &mut other), E_INVALIDARG, "misaligned");
    // The resource keeps the heap alive.
    assert_eq!(slot::<Release>(heap, 0x10)(heap), 1);
    assert_eq!(slot::<Release>(placed, 0x10)(placed), 0);
}

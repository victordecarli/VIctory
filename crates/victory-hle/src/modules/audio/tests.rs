use super::*;

#[test]
fn heap_maps_every_byte_of_a_block() {
    let _serial = lock(&CONTEXT_TESTS);
    let (mut ptr, mut apu) = (std::ptr::null_mut(), 0);
    assert_eq!(ApuAlloc(&mut ptr, &mut apu, 0x100, 128, 1), S_OK);
    assert!(apu != 0 && apu.is_multiple_of(128));
    let inner = ptr.cast::<u8>().wrapping_add(0x40).cast_const().cast();
    assert_eq!(ApuMapVirtualAddress(inner), apu + 0x40);
    assert!(ApuIsVirtualAddressValid(ptr, 128));
    assert_eq!(heap_unmap(apu + 4), Some(ptr.cast::<u8>().wrapping_add(4)));
    ApuFree(ptr);
    assert_eq!(ApuMapVirtualAddress(ptr), 0);
    assert_eq!(ApuAlloc(&mut ptr, &mut apu, 0x100, 6, 1), E_INVALIDARG);
}

/// The context arrays and the heap exist once per process: tests that
/// allocate from them run one at a time (a concurrent allocation could
/// take a freed range a test expects back).
static CONTEXT_TESTS: Mutex<()> = Mutex::new(());

#[test]
fn heap_reuses_freed_addresses() {
    let _serial = lock(&CONTEXT_TESTS);
    let (mut a, mut b, mut c) = (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
    let (mut apu_a, mut apu_b, mut apu_c) = (0, 0, 0);
    assert_eq!(ApuAlloc(&mut a, &mut apu_a, 0x1000, 0x1000, 1), S_OK);
    assert_eq!(ApuAlloc(&mut b, &mut apu_b, 0x1000, 0x1000, 1), S_OK);
    ApuFree(a);
    assert_eq!(ApuAlloc(&mut c, &mut apu_c, 0x800, 0x800, 1), S_OK);
    // The freed range is free again, whichever part `c` took.
    assert!(
        lock(&HEAP).free.iter().any(|(&s, &e)| s <= apu_a + 0x800 && apu_a + 0x1000 <= e)
            || apu_c == apu_a
    );
    assert_eq!(ApuMapApuAddress(apu_b + 8), b.cast::<u8>().wrapping_add(8).cast());
    assert!(ApuMapApuAddress(0).is_null());
    assert_eq!(ApuCreateHeap(64 << 20, 0), 0x8ACA_0001_u32 as i32, "too late");
    ApuFree(b);
    ApuFree(c);
}

/// Two packets of three silent mono frames each: a fixed one-subframe
/// layout, no transmitted coefficients.
fn silent_stream() -> Vec<u8> {
    let mut data = vec![0xffu8; 2 * xma::PACKET_BYTES];
    let mut put = |at: usize, value: u32, width: usize| {
        for i in 0..width {
            let (p, bit) = (at + i, (value >> (width - 1 - i)) & 1);
            data[p / 8] = (data[p / 8] & !(0x80 >> (p % 8))) | ((bit as u8) << (7 - p % 8));
        }
    };
    for packet in 0..2 {
        let base = packet * xma::PACKET_BYTES * 8;
        put(base, 3 << 26, 32);
        for frame in 0..3 {
            let at = base + 32 + frame * 31;
            put(at, 31, 15); // length
            put(at + 15, 0b10, 2); // fixed layout, one 512-sample subframe
            put(at + 17, 0, 8 + 1 + 3 + 2); // gain, no trim, empty subframe
        }
    }
    data
}

#[test]
fn xma_contexts_decode_into_their_output_rings() {
    let _serial = lock(&CONTEXT_TESTS);
    let mut request = ShapeContexts {
        counts: [0, 0, 0, 0, 4, 0],
        arrays: [std::ptr::null_mut(); 6],
        apu: [0; 6],
    };
    assert_eq!(AcpHalAllocateShapeContexts(&mut request), S_OK);
    let context = Context(request.arrays[4].cast::<u32>().wrapping_add(16));

    let (mut input, mut input_apu) = (std::ptr::null_mut(), 0);
    assert_eq!(ApuAlloc(&mut input, &mut input_apu, 0x1000, 0x800, 1), S_OK);
    let stream = silent_stream();
    // SAFETY: two packets were allocated.
    unsafe { std::ptr::copy_nonoverlapping(stream.as_ptr(), input.cast(), stream.len()) };
    let (mut output, mut output_apu) = (std::ptr::null_mut(), 0);
    assert_eq!(ApuAlloc(&mut output, &mut output_apu, 0x400, 0x100, 2), S_OK);
    // SAFETY: the ring was just allocated.
    unsafe { output.cast::<u8>().write_bytes(0x55, 0x400) };

    // As the title sets one up: two packets, input 0 valid, a ring of
    // four blocks, writable, 48 kHz mono, reading from the first frame.
    context.set(0, 2 | 1 << 20 | 4 << 22);
    context.set(1, 1 << 31 | 3 << 27);
    context.set(2, 32);
    context.set(5, input_apu);
    context.set(7, output_apu);

    let mut acp = std::ptr::null_mut();
    assert_eq!(AcpHalCreate(&mut acp), S_OK);
    // SAFETY: just created.
    let m = unsafe { (*acp).vtable };
    assert_eq!((m.connect)(acp, 0x1000, 0x1000), S_OK);
    let submit = |command: u32, words: &[u32]| {
        assert_eq!((m.submit)(acp, command, 0xf1, u32::MAX, words.as_ptr().cast(), 0), S_OK);
    };
    submit(ENABLE_XMA, &[1]);
    xma().pump();
    // Full: all four blocks written (silence), the write block wrapped
    // onto the read block, and the ring no longer writable.
    assert_eq!(context.field(0, 27, 5), 0);
    assert_eq!(context.field(1, 31, 1), 0);
    // SAFETY: the ring.
    let ring = unsafe { std::slice::from_raw_parts(output.cast::<u8>(), 0x400) };
    assert!(ring.iter().all(|&b| b == 0));

    // Read four blocks at a time until the input has run out and the
    // ring keeps room (the decoding thread may fill it too).
    for _ in 0..20 {
        submit(ADVANCE_XMA_READ, &[1, 4]);
        xma().pump();
        if context.field(2, 31, 1) == 1 && context.field(1, 31, 1) == 1 {
            break;
        }
    }
    // Six frames and the flushed tail: 3,328 samples, 26 blocks.
    let produced = |x: &Xma| lock(&x.voices[1]).as_ref().map(|v| v.trace.produced);
    assert_eq!(produced(&xma()), Some(26 * SAMPLES_PER_BLOCK));
    assert_eq!(context.field(0, 20, 2), 0, "input used up");
    assert_eq!(context.field(2, 26, 5), INPUT_INVALID);

    // A count near `u32::MAX` wraps the read block like any other
    // (3 + 0xffff_ffff blocks on a ring of four is block 2); the sum
    // used to overflow.
    context.set_field(9, 0, 5, 3);
    submit(ADVANCE_XMA_READ, &[1, u32::MAX]);
    assert_eq!(context.field(9, 0, 5), 2);

    submit(DISABLE_XMA, &[1]);
    assert_eq!(produced(&xma()), None);
    (m.release)(acp);
    ApuFree(input);
    ApuFree(output);
    assert_eq!(AcpHalReleaseShapeContexts(), S_OK);
}

#[test]
fn full_message_queues_count_drops() {
    let mut acp = std::ptr::null_mut();
    assert_eq!(AcpHalCreate(&mut acp), S_OK);
    // SAFETY: just created.
    let m = unsafe { (*acp).vtable };
    assert_eq!((m.connect)(acp, 16, 2), S_OK);
    let mask = COMMAND_COMPLETED;
    assert_eq!((m.submit)(acp, REGISTER_MESSAGES, 0, u32::MAX, (&raw const mask).cast(), 0), S_OK);
    for id in 1..=4 {
        assert_eq!((m.submit)(acp, 12, id, u32::MAX, std::ptr::null(), 0), S_OK);
    }
    assert_eq!((m.message_count)(acp), 2);
    let mut message = Message { kind: 0, dropped: 0, usec: 0, data: [0; 16] };
    assert!((m.pop_message)(acp, &mut message) && (m.pop_message)(acp, &mut message));
    assert_eq!((m.submit)(acp, 12, 5, u32::MAX, std::ptr::null(), 0), S_OK);
    assert!((m.pop_message)(acp, &mut message));
    let dropped = message.dropped;
    assert_eq!(dropped, 2);
    (m.release)(acp);
}

#[test]
fn startup_as_fmod_does_it() {
    let _serial = lock(&CONTEXT_TESTS);
    // Context arrays: rate conversion, DMA and XMA.
    let mut request = ShapeContexts {
        counts: [0x200, 0, 0, 0x400, 0x200, 0],
        arrays: [std::ptr::null_mut(); 6],
        apu: [0; 6],
    };
    assert_eq!(size_of::<ShapeContexts>(), 0x60);
    assert_eq!(AcpHalAllocateShapeContexts(&mut request), S_OK);
    let arrays = request.arrays;
    assert!(!arrays[0].is_null() && !arrays[3].is_null() && !arrays[4].is_null());
    assert!(arrays[1].is_null());
    // SAFETY: 0x200 rate-conversion contexts of 32 bytes were allocated.
    unsafe { arrays[0].cast::<u8>().write_bytes(0, 0x200 << 5) };

    // Connect, register every message but frame starts, then wait for
    // the registration to be acknowledged.
    let mut acp = std::ptr::null_mut();
    assert_eq!(AcpHalCreate(&mut acp), S_OK);
    // SAFETY: just created.
    let m = unsafe { (*acp).vtable };
    assert_eq!((m.connect)(acp, 0x804, 0x804), S_OK);
    let mask = 0xfe_u32;
    let data = (&raw const mask).cast();
    assert_eq!((m.submit)(acp, REGISTER_MESSAGES, 1, u32::MAX, data, 0), S_OK);
    let mut message = Message { kind: 0, dropped: 0, usec: 0, data: [0; 16] };
    assert!((m.pop_message)(acp, &mut message));
    let (kind, dropped) = (message.kind, message.dropped);
    assert_eq!((kind, dropped), (COMMAND_COMPLETED, 0));
    assert_eq!(u64::from_le_bytes(message.data[4..12].try_into().unwrap()), 1);
    assert!(!(m.pop_message)(acp, &mut message));
    assert_eq!((m.disconnect)(acp), S_OK);
    assert_eq!((m.message_count)(acp), 1, "disconnected");
    (m.release)(acp);
    assert_eq!(AcpHalReleaseShapeContexts(), S_OK);
}

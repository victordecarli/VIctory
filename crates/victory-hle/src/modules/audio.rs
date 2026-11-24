//! The console's audio hardware (`AcpHal.dll`):
//!
//! * the APU heap: host memory with 32-bit device addresses;
//! * SHAPE context arrays allocated from it;
//! * the audio coprocessor's command and message queues, where every
//!   command completes at once and is acknowledged as the hardware would;
//! * the XMA decoder: enabled XMA contexts are decoded on a thread of ours
//!   into their output rings, like the hardware does.
//!
//! SHAPE's other units (rate conversion, EQ, filters, mixing through
//! flowgraphs) are not emulated, so voices that need them stay silent.
//! Mixed audio leaves through the host's WASAPI, which titles and
//! middleware reach through COM like on PC. fmod's console build sets up
//! the coprocessor inside `System_Create` and waits there for its first
//! command to be acknowledged; Grand Theft Auto VI uses only XMA contexts.
#![allow(
    non_snake_case,
    reason = "exports carry the exact API names (`ApuCreateHeap`) that `hle_export!` binds imports by"
)]

use std::alloc::Layout;
use std::collections::{BTreeMap, VecDeque};
use std::ffi::c_void;
use std::sync::{Mutex, MutexGuard, OnceLock, RwLock, RwLockReadGuard};
use std::time::Instant;

use super::xasync::E_INVALIDARG;
use super::xgameruntime::{HRESULT, S_OK};
use crate::{HleModule, hle_export};

#[cfg(all(windows, feature = "diagnostics"))]
pub mod capture;

const E_FAIL: HRESULT = 0x8000_4005_u32 as i32;
const E_OUTOFMEMORY: HRESULT = 0x8007_000E_u32 as i32;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// The APU heap.

struct Block {
    size: usize,
    apu: u32,
    layout: Layout,
}

struct Heap {
    /// By host address.
    blocks: BTreeMap<usize, Block>,
    /// Host address by device address.
    by_apu: BTreeMap<u32, usize>,
    /// Free device address ranges, start to end, merged with neighbours.
    free: BTreeMap<u32, u32>,
    /// Set by `ApuCreateHeap` or the first allocation: the heap exists.
    created: bool,
}

static HEAP: Mutex<Heap> = Mutex::new(Heap {
    blocks: BTreeMap::new(),
    by_apu: BTreeMap::new(),
    free: BTreeMap::new(),
    created: false,
});

/// Device addresses start above 0, which means "none" to titles.
const APU_BASE: u32 = 0x0001_0000;

impl Heap {
    /// First fit in the free ranges.
    fn take(&mut self, size: u32, align: u32) -> Option<u32> {
        if self.free.is_empty() && !self.created {
            self.free.insert(APU_BASE, u32::MAX);
        }
        self.created = true;
        let (start, end, apu) = self.free.iter().find_map(|(&start, &end)| {
            let apu = start.checked_next_multiple_of(align)?;
            (apu.checked_add(size)? <= end).then_some((start, end, apu))
        })?;
        self.free.remove(&start);
        if start < apu {
            self.free.insert(start, apu);
        }
        if apu + size < end {
            self.free.insert(apu + size, end);
        }
        Some(apu)
    }

    fn give_back(&mut self, mut start: u32, mut end: u32) {
        if let Some((&before, &before_end)) = self.free.range(..start).next_back()
            && before_end == start
        {
            self.free.remove(&before);
            start = before;
        }
        if let Some(after_end) = self.free.remove(&end) {
            end = after_end;
        }
        self.free.insert(start, end);
    }
}

/// Zeroed host memory and its device address.
fn heap_alloc(size: usize, align: usize) -> Option<(*mut u8, u32)> {
    let layout = Layout::from_size_align(size.max(1), align.max(16)).ok()?;
    let mut heap = lock(&HEAP);
    let apu = heap.take(u32::try_from(layout.size()).ok()?, align as u32)?;
    // SAFETY: non-zero size.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        heap.give_back(apu, apu + layout.size() as u32);
        return None;
    }
    heap.blocks.insert(ptr as usize, Block { size, apu, layout });
    heap.by_apu.insert(apu, ptr as usize);
    Some((ptr, apu))
}

fn heap_free(ptr: *mut u8) {
    let mut heap = lock(&HEAP);
    let Some(block) = heap.blocks.remove(&(ptr as usize)) else {
        return;
    };
    heap.by_apu.remove(&block.apu);
    heap.give_back(block.apu, block.apu + block.layout.size() as u32);
    drop(heap);
    // SAFETY: allocated by `heap_alloc` with this layout.
    unsafe { std::alloc::dealloc(ptr, block.layout) };
}

/// Sizes the heap once, before anything is allocated from it. The console
/// reserves both parts up front; we allocate per block, so the sizes only
/// decide whether the call is valid.
// shortcut: the cached and non-cached sizes are not enforced as limits;
// enforce them if a title relies on running out.
pub extern "system" fn ApuCreateHeap(cached: u32, _non_cached: u32) -> HRESULT {
    const ALREADY_ALLOCATED: HRESULT = 0x8ACA_0001_u32 as i32;
    if cached == 0 {
        return E_INVALIDARG;
    }
    let mut heap = lock(&HEAP);
    if heap.created {
        return ALREADY_ALLOCATED;
    }
    heap.created = true;
    heap.free.insert(APU_BASE, u32::MAX);
    S_OK
}

/// The host address of a device address, null outside the heap.
pub extern "system" fn ApuMapApuAddress(apu: u32) -> *mut c_void {
    heap_unmap(apu).map_or(std::ptr::null_mut(), |p| p.cast())
}

/// The device address of any byte of a heap block, 0 elsewhere.
fn heap_map(ptr: usize) -> u32 {
    let heap = lock(&HEAP);
    match heap.blocks.range(..=ptr).next_back() {
        Some((&start, block)) if ptr < start + block.size.max(1) => {
            block.apu + (ptr - start) as u32
        }
        _ => 0,
    }
}

/// The host address of a device address, if a heap block holds it.
fn heap_unmap(apu: u32) -> Option<*mut u8> {
    let heap = lock(&HEAP);
    let (&start, &host) = heap.by_apu.range(..=apu).next_back()?;
    let block = &heap.blocks[&host];
    (apu < start.saturating_add(block.size.max(1) as u32))
        .then(|| (host + (apu - start) as usize) as *mut u8)
}

/// `len` bytes of title memory at a device address, if one heap block holds
/// them all.
fn heap_bytes<'a>(apu: u32, len: usize) -> Option<&'a mut [u8]> {
    let heap = lock(&HEAP);
    let (&start, &host) = heap.by_apu.range(..=apu).next_back()?;
    let offset = (apu - start) as usize;
    (offset + len <= heap.blocks[&host].size)
        // SAFETY: inside a live heap block, which the title and the audio
        // hardware share like on the console.
        .then(|| unsafe { std::slice::from_raw_parts_mut((host + offset) as *mut u8, len) })
}

/// # Safety
///
/// `out` is null or valid for a write of a pointer, and `apu` is null or
/// valid for a write of a `u32`.
pub unsafe extern "system" fn ApuAlloc(
    out: *mut *mut c_void,
    apu: *mut u32,
    size: u32,
    align: u32,
    _flags: u32,
) -> HRESULT {
    if out.is_null() || size == 0 || align < 4 || !align.is_power_of_two() {
        return E_INVALIDARG;
    }
    let Some((ptr, address)) = heap_alloc(size as usize, align as usize) else {
        return E_OUTOFMEMORY;
    };
    // SAFETY: checked non-null; writable, per this function's contract; the
    // optional `apu` is checked below.
    unsafe {
        out.write(ptr.cast());
        if !apu.is_null() {
            apu.write(address);
        }
    }
    S_OK
}

pub extern "system" fn ApuFree(ptr: *mut c_void) {
    heap_free(ptr.cast());
}

pub extern "system" fn ApuMapVirtualAddress(ptr: *const c_void) -> u32 {
    heap_map(ptr as usize)
}

pub extern "system" fn ApuIsVirtualAddressValid(ptr: *const c_void, align: u32) -> bool {
    let apu = heap_map(ptr as usize);
    apu != 0 && (align == 0 || apu.is_multiple_of(align))
}

// SHAPE context arrays.

/// Six counts (rate conversion, EQ/compression, filter/volume, DMA, XMA,
/// PCM), then the six arrays and their device addresses.
#[repr(C, packed)]
pub struct ShapeContexts {
    counts: [u32; 6],
    arrays: [*mut c_void; 6],
    apu: [u32; 6],
}

/// Bytes per context: rate conversion, DMA and XMA as fmod clears them;
/// the others (not seen yet) get a generous upper bound.
const CONTEXT_BYTES: [usize; 6] = [32, 256, 256, 16, 64, 256];

static CONTEXTS: Mutex<Option<[usize; 6]>> = Mutex::new(None);

/// # Safety
///
/// `contexts` is null or valid for reads and writes of a `ShapeContexts`,
/// unaligned.
pub unsafe extern "system" fn AcpHalAllocateShapeContexts(contexts: *mut ShapeContexts) -> HRESULT {
    if contexts.is_null() {
        return E_INVALIDARG;
    }
    let mut allocated = lock(&CONTEXTS);
    if allocated.is_some() {
        return E_FAIL;
    }
    // SAFETY: checked non-null; readable, per this function's contract;
    // packed fields are copied out.
    let counts = unsafe { std::ptr::addr_of!((*contexts).counts).read_unaligned() };
    let mut arrays: [*mut c_void; 6] = [std::ptr::null_mut(); 6];
    let mut addresses = [0u32; 6];
    for i in 0..6 {
        if counts[i] == 0 {
            continue;
        }
        let Some((ptr, apu)) = heap_alloc(counts[i] as usize * CONTEXT_BYTES[i], 256) else {
            arrays.iter().for_each(|&p| heap_free(p.cast()));
            return E_OUTOFMEMORY;
        };
        (arrays[i], addresses[i]) = (ptr.cast(), apu);
    }
    // SAFETY: as above; writable too.
    unsafe {
        std::ptr::addr_of_mut!((*contexts).arrays).write_unaligned(arrays);
        std::ptr::addr_of_mut!((*contexts).apu).write_unaligned(addresses);
    }
    *allocated = Some(arrays.map(|p| p as usize));
    let mut xma = XMA.write().unwrap_or_else(|e| e.into_inner());
    xma.array = arrays[4] as usize;
    xma.voices = (0..counts[4]).map(|_| Mutex::new(None)).collect();
    S_OK
}

pub extern "system" fn AcpHalReleaseShapeContexts() -> HRESULT {
    match lock(&CONTEXTS).take() {
        Some(arrays) => {
            *XMA.write().unwrap_or_else(|e| e.into_inner()) = Xma::EMPTY;
            arrays.iter().for_each(|&p| heap_free(p as *mut u8));
            S_OK
        }
        None => E_FAIL,
    }
}

// The XMA decoder.
//
// An XMA context (16 words in the shared context array) names up to two
// input buffers of 2 KiB packets, a ring of 256-byte output blocks the
// hardware fills with 16-bit PCM, and the read offset the title advances
// by command. Words and bit fields, as titles write them:
// 0: input 0 size in packets (0-11), loop count (12-19, 255 forever),
//    valid inputs (20-21), output size in blocks (22-26), write block
//    (27-31, hardware);
// 1: input 1 size (0-11), loop end subframe (12-13), loop start subframes
//    to skip (17-19), subframes to skip at the start (24-26), sample rate
//    (27-28: 24, 32, 44.1, 48 kHz), stereo (29), output writable (31);
// 2: read position in bits (0-25), error status (26-30), error (31);
// 3: loop start in bits (0-25); 4: loop end in bits (0-25), current input
//    (31); 5, 6: input addresses; 7: output address; 8: overlap buffer;
//    9: read block (0-4).
// Titles read back only the write block, "writable" and the read block,
// and take whatever lies between read and write.
// shortcut: each enabled context decodes from a cold state and the overlap
// buffer is ignored (titles copy it between consecutive contexts of one
// voice); keep decoder state there if a title's seams turn out audible.

use victory_formats::xma;

const ENABLE_XMA: u32 = 5;
const ENABLE_XMAS: u32 = 6;
const DISABLE_XMA: u32 = 7;
const DISABLE_XMAS: u32 = 8;
const ADVANCE_XMA_READ: u32 = 18;

const BLOCK_BYTES: usize = 256;
const SAMPLES_PER_BLOCK: usize = BLOCK_BYTES / 2;
/// Error status: the input ran out.
const INPUT_INVALID: u32 = 4;

struct Voice {
    decoder: Option<xma::Decoder>,
    pcm: VecDeque<i16>,
    /// Samples per channel still to drop (start or loop subframe skips).
    skip: usize,
    ended: bool,
    trace: Trace,
}

/// What a voice did, for the debug log when it is disabled.
struct Trace {
    since: Instant,
    /// Samples per channel written to the ring.
    produced: usize,
    /// Blocks the title took (command 18), how often, and how often it
    /// found the ring less than full.
    taken: usize,
    takes: u32,
    partial_takes: u32,
    frames: u32,
    bad_frames: u32,
    /// The loudest sample written.
    peak: i16,
    /// Milliseconds after enabling when the input ran out.
    ended_ms: Option<u128>,
}

struct Xma {
    /// Host address of the context array.
    array: usize,
    /// One entry per context, `Some` while enabled. Each has its own lock,
    /// so a command for one voice never waits for others to decode (the
    /// title sends them from its time-critical mixer thread).
    voices: Vec<Mutex<Option<Voice>>>,
}

impl Xma {
    const EMPTY: Xma = Xma { array: 0, voices: Vec::new() };
}

/// Written only when the context arrays are allocated or released.
static XMA: RwLock<Xma> = RwLock::new(Xma::EMPTY);

fn xma() -> RwLockReadGuard<'static, Xma> {
    XMA.read().unwrap_or_else(|e| e.into_inner())
}

/// A context's words in title memory.
#[derive(Clone, Copy)]
struct Context(*mut u32);

impl Context {
    fn get(self, word: usize) -> u32 {
        // SAFETY: inside the context array; the title reads and writes it
        // concurrently, like the hardware's.
        unsafe { self.0.add(word).read_volatile() }
    }

    fn set(self, word: usize, value: u32) {
        // SAFETY: as above.
        unsafe { self.0.add(word).write_volatile(value) }
    }

    fn field(self, word: usize, shift: u32, bits: u32) -> u32 {
        (self.get(word) >> shift) & ((1 << bits) - 1)
    }

    fn set_field(self, word: usize, shift: u32, bits: u32, value: u32) {
        let mask = ((1u32 << bits) - 1) << shift;
        self.set(word, (self.get(word) & !mask) | ((value << shift) & mask));
    }

    fn channels(self) -> usize {
        1 + self.field(1, 29, 1) as usize
    }
}

impl Xma {
    fn context(&self, index: u32) -> Option<Context> {
        ((index as usize) < self.voices.len() && self.array != 0)
            .then(|| Context((self.array + index as usize * 64) as *mut u32))
    }

    fn enable(&self, index: u32) {
        let Some(context) = self.context(index) else {
            return;
        };
        let mut slot = lock(&self.voices[index as usize]);
        context.set_field(2, 26, 6, 0);
        tracing::debug!(
            index,
            words = ?(0..10).map(|w| format!("{:08x}", context.get(w))).collect::<Vec<_>>(),
            "XMA voice enabled"
        );
        #[cfg(feature = "diagnostics")]
        dump_input(index, context);
        *slot = Some(Voice {
            decoder: None,
            pcm: VecDeque::new(),
            skip: context.field(1, 24, 3) as usize * xma::SUBFRAME_SAMPLES,
            ended: false,
            trace: Trace {
                since: Instant::now(),
                produced: 0,
                taken: 0,
                takes: 0,
                partial_takes: 0,
                frames: 0,
                bad_frames: 0,
                peak: 0,
                ended_ms: None,
            },
        });
        drop(slot);
        wake_decoder();
    }

    fn disable(&self, index: u32) {
        let Some(context) = self.context(index) else {
            return;
        };
        let Some(voice) = lock(&self.voices[index as usize]).take() else {
            return;
        };
        let t = &voice.trace;
        tracing::debug!(
            index,
            ms = t.since.elapsed().as_millis(),
            produced = t.produced,
            taken = t.taken * SAMPLES_PER_BLOCK / context.channels(),
            takes = t.takes,
            partial_takes = t.partial_takes,
            frames = t.frames,
            bad_frames = t.bad_frames,
            peak = t.peak,
            input_ended_ms = ?t.ended_ms,
            read_bit = context.field(2, 0, 26),
            input_bits = context.field(0, 0, 12) * xma::PACKET_BYTES as u32 * 8,
            "XMA voice disabled"
        );
    }

    /// Moves the title's read block on, which frees output space.
    fn advance_read(&self, index: u32, blocks: u32) {
        let Some(context) = self.context(index) else {
            return;
        };
        // The decoder writes the same words.
        let mut slot = lock(&self.voices[index as usize]);
        let size = context.field(0, 22, 5).max(1);
        // The count is the title's: reduced first, so the sum stays small.
        let read = (context.field(9, 0, 5) + blocks % size) % size;
        context.set_field(9, 0, 5, read);
        if blocks > 0 {
            context.set_field(1, 31, 1, 1);
        }
        if let Some(voice) = &mut *slot {
            let t = &mut voice.trace;
            t.taken += blocks as usize;
            t.takes += 1;
            t.partial_takes += u32::from(blocks < size);
        }
        drop(slot);
        wake_decoder();
    }

    /// Fills every enabled context's output as far as it has room.
    fn pump(&self) {
        let start = Instant::now();
        let mut active = 0;
        for index in 0..self.voices.len() as u32 {
            if let (Some(context), Some(voice)) =
                (self.context(index), &mut *lock(&self.voices[index as usize]))
            {
                pump(context, voice);
                active += 1;
            }
        }
        let took = start.elapsed();
        if took.as_millis() >= 4 {
            tracing::debug!(active, ms = took.as_millis(), "slow XMA decoding pass");
        }
    }
}

fn pump(context: Context, voice: &mut Voice) {
    let size = context.field(0, 22, 5);
    if size == 0 {
        return;
    }
    while context.field(1, 31, 1) == 1 {
        if voice.pcm.len() < SAMPLES_PER_BLOCK && !decode(context, voice) {
            return;
        }
        let write = context.field(0, 27, 5);
        let address = context.get(7) as usize + write as usize * BLOCK_BYTES;
        let Some(out) = heap_bytes(address as u32, BLOCK_BYTES) else {
            return;
        };
        for pair in out.as_chunks_mut::<2>().0 {
            let sample = voice.pcm.pop_front().unwrap_or(0);
            voice.trace.peak = voice.trace.peak.max(sample.saturating_abs());
            pair.copy_from_slice(&sample.to_le_bytes());
        }
        voice.trace.produced += SAMPLES_PER_BLOCK / context.channels();
        let write = (write + 1) % size;
        context.set_field(0, 27, 5, write);
        if write == context.field(9, 0, 5) {
            // Full: nothing more until the title advances its read block.
            context.set_field(1, 31, 1, 0);
        }
    }
}

/// Decodes the next frame of the current input into `voice.pcm`; false
/// when there is nothing left to decode.
fn decode(context: Context, voice: &mut Voice) -> bool {
    if voice.ended {
        return false;
    }
    let channels = context.channels();
    let mut current = context.field(4, 31, 1);
    let valid = context.field(0, 20, 2);
    if valid & (1 << current) == 0 {
        if valid & (1 << (1 - current)) == 0 {
            // Out of input: the overlap tail, then nothing.
            if let Some(decoder) = &mut voice.decoder {
                let mut tail = Vec::new();
                decoder.flush(&mut tail);
                voice.pcm.extend(tail);
            }
            voice.ended = true;
            voice.trace.ended_ms = Some(voice.trace.since.elapsed().as_millis());
            context.set_field(2, 26, 5, INPUT_INVALID);
            context.set_field(2, 31, 1, 1);
            return !voice.pcm.is_empty();
        }
        current = 1 - current;
        context.set_field(4, 31, 1, current);
        context.set_field(2, 0, 26, 0);
    }
    let (address, packets) = match current {
        0 => (context.get(5), context.field(0, 0, 12)),
        _ => (context.get(6), context.field(1, 0, 12)),
    };
    let Some(data) = heap_bytes(address & !0x7ff, packets as usize * xma::PACKET_BYTES) else {
        context.set_field(0, 20 + current, 1, 0);
        return true;
    };
    let mut at = context.field(2, 0, 26) as usize;
    if at < 32 {
        at = 32 + xma::packet(data, 0).map_or(0, |p| p.first_frame_bit as usize);
    }
    let Some(frame) = xma::read_frame(data, at) else {
        // This input is used up.
        context.set_field(0, 20 + current, 1, 0);
        return true;
    };
    let decoder = voice.decoder.get_or_insert_with(|| {
        const RATES: [u32; 4] = [24_000, 32_000, 44_100, 48_000];
        xma::Decoder::new(channels, RATES[context.field(1, 27, 2) as usize])
    });
    let mut pcm = Vec::with_capacity(xma::FRAME_SAMPLES * channels);
    voice.trace.frames += 1;
    if let Err(e) = decoder.decode(&frame.data, frame.bits, &mut pcm) {
        voice.trace.bad_frames += 1;
        if voice.trace.bad_frames <= 2 {
            tracing::debug!(
                at,
                bits = frame.bits,
                head = ?&frame.data[..frame.data.len().min(8)],
                %e,
                "XMA frame did not decode"
            );
        }
    }
    let mut next = frame.next.unwrap_or(0);
    let loops = context.field(0, 12, 8);
    let loop_end = context.field(4, 0, 26) as usize;
    let ends_loop = loop_end != 0
        && (frame.start == loop_end
            || (frame.start < loop_end && frame.next.is_none_or(|n| loop_end < n)));
    if loops > 0 && ends_loop {
        let keep = (context.field(1, 12, 2) as usize + 1) * xma::SUBFRAME_SAMPLES * channels;
        pcm.truncate(keep);
        next = context.field(3, 0, 26) as usize;
        voice.skip = context.field(1, 17, 3) as usize * xma::SUBFRAME_SAMPLES;
        if loops != 255 {
            context.set_field(0, 12, 8, loops - 1);
        }
    }
    let skip = (voice.skip * channels).min(pcm.len());
    voice.skip -= skip / channels;
    voice.pcm.extend(&pcm[skip..]);
    if next == 0 {
        context.set_field(0, 20 + current, 1, 0);
    }
    context.set_field(2, 0, 26, next as u32);
    true
}

/// `VICTORY_XMA_DUMP=<dir>`: saves the context words (`<n>-<index>.ctx`)
/// and input buffer 0 (`.xma`) of the first 32 voices enabled, to test
/// the decoder on a title's streams outside the title.
#[cfg(feature = "diagnostics")]
fn dump_input(index: u32, context: Context) {
    static COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let Some(dir) = std::env::var_os("VICTORY_XMA_DUMP") else {
        return;
    };
    let n = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n >= 32 {
        return;
    }
    let dir = std::path::PathBuf::from(dir);
    let words: Vec<u8> = (0..16).flat_map(|w| context.get(w).to_le_bytes()).collect();
    let input =
        heap_bytes(context.get(5) & !0x7ff, context.field(0, 0, 12) as usize * xma::PACKET_BYTES);
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(dir.join(format!("{n:02}-{index}.ctx")), words);
    let _ = std::fs::write(dir.join(format!("{n:02}-{index}.xma")), input.unwrap_or_default());
}

/// Starts the decoding thread on the first enabled voice, and has it fill
/// rings now: a voice just enabled or read from.
fn wake_decoder() {
    static DECODER: OnceLock<std::thread::Thread> = OnceLock::new();
    DECODER
        .get_or_init(|| {
            std::thread::Builder::new()
                .name("victory XMA decoder".into())
                .spawn(|| {
                    loop {
                        xma().pump();
                        // A 256-byte output block lasts 2.7 ms at 48 kHz mono.
                        std::thread::park_timeout(std::time::Duration::from_millis(2));
                    }
                })
                .map_or_else(|_| std::thread::current(), |h| h.thread().clone())
        })
        .unpark();
}

/// Runs an XMA command; false for one this does not know.
fn xma_command(command: u32, word: impl Fn(usize) -> Option<u32>) -> bool {
    let xma = xma();
    match command {
        ENABLE_XMA => xma.enable(word(0).unwrap_or(u32::MAX)),
        DISABLE_XMA => xma.disable(word(0).unwrap_or(u32::MAX)),
        ENABLE_XMAS | DISABLE_XMAS => {
            for i in 0..xma.voices.len() as u32 {
                if word(i as usize / 32).unwrap_or(0) & (1 << (i % 32)) != 0 {
                    match command {
                        ENABLE_XMAS => xma.enable(i),
                        _ => xma.disable(i),
                    }
                }
            }
        }
        ADVANCE_XMA_READ => xma.advance_read(word(0).unwrap_or(u32::MAX), word(1).unwrap_or(0)),
        _ => return false,
    }
    true
}

// The audio coprocessor.

const LOAD_FLOWGRAPH: u32 = 1;
const REGISTER_MESSAGES: u32 = 2;
const UNREGISTER_MESSAGES: u32 = 3;

const FLOWGRAPH_COMPLETED: u32 = 0x02;
const COMMAND_COMPLETED: u32 = 0x10;
const DISCONNECTED: u32 = 0x80;

/// Audio frames of 128 samples at 48 kHz.
const FRAMES_PER_SECOND: u64 = 375;

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct Message {
    kind: u32,
    dropped: u32,
    usec: u32,
    data: [u8; 16],
}

struct State {
    connected: bool,
    /// Message kinds the title registered for.
    mask: u32,
    messages: VecDeque<Message>,
    /// Queue size given to `connect`.
    capacity: usize,
    /// Messages lost to a full queue since the last one queued.
    dropped: u32,
    since: Instant,
}

impl State {
    fn usec(&self) -> u64 {
        self.since.elapsed().as_micros() as u64
    }

    fn frame(&self) -> u32 {
        (self.usec() * FRAMES_PER_SECOND / 1_000_000) as u32
    }

    /// Queues a registered message; a full queue drops it and counts it in
    /// the next one that fits.
    fn post(&mut self, kind: u32, data: &[u8]) {
        if self.mask & kind == 0 {
            return;
        }
        if self.messages.len() >= self.capacity {
            self.dropped = self.dropped.saturating_add(1);
            tracing::debug!(kind, dropped = self.dropped, "coprocessor message dropped");
            return;
        }
        let mut message = Message {
            kind,
            dropped: std::mem::take(&mut self.dropped),
            usec: self.usec() as u32,
            data: [0; 16],
        };
        message.data[..data.len()].copy_from_slice(data);
        self.messages.push_back(message);
    }
}

/// The interface object: its method table, then our state.
#[repr(C)]
pub struct Acp {
    vtable: &'static AcpMethods,
    state: Mutex<State>,
}

#[repr(C)]
struct AcpMethods {
    connect: unsafe extern "system" fn(*mut Acp, u32, u32) -> HRESULT,
    disconnect: unsafe extern "system" fn(*mut Acp) -> HRESULT,
    submit: unsafe extern "system" fn(*mut Acp, u32, u64, u32, *const c_void, u32) -> HRESULT,
    pop_message: unsafe extern "system" fn(*mut Acp, *mut Message) -> bool,
    message_count: unsafe extern "system" fn(*mut Acp) -> u32,
    release: unsafe extern "system" fn(*mut Acp),
}

static ACP_METHODS: AcpMethods =
    AcpMethods { connect, disconnect, submit, pop_message, message_count, release };

/// # Safety
///
/// `this` is the live `Acp` `AcpHalCreate` made, which lives until
/// `release`.
unsafe fn state(this: *mut Acp) -> MutexGuard<'static, State> {
    // SAFETY: `this` is a live `Acp`, per this function's contract.
    lock(unsafe { &(*this).state })
}

/// # Safety
///
/// `this` is the live `Acp` `AcpHalCreate` made.
unsafe extern "system" fn connect(this: *mut Acp, commands: u32, messages: u32) -> HRESULT {
    if commands == 0 {
        return E_INVALIDARG;
    }
    // SAFETY: `this` as this function's contract gives it.
    let mut state = unsafe { state(this) };
    state.connected = true;
    state.capacity = messages as usize;
    S_OK
}

/// # Safety
///
/// `this` is the live `Acp` `AcpHalCreate` made.
unsafe extern "system" fn disconnect(this: *mut Acp) -> HRESULT {
    // SAFETY: `this` as this function's contract gives it.
    let mut state = unsafe { state(this) };
    if !state.connected {
        return E_FAIL;
    }
    state.post(DISCONNECTED, &[]);
    state.connected = false;
    S_OK
}

/// Done at once: message registrations take effect, a flowgraph completes
/// unprocessed, a command with an id is acknowledged, and its notification
/// word is set.
///
/// # Safety
///
/// `this` is the live `Acp` `AcpHalCreate` made. `data` is null or valid
/// for reads of the command's words, as many as it reads (up to the
/// command's size).
unsafe extern "system" fn submit(
    this: *mut Acp,
    command: u32,
    id: u64,
    _frame: u32,
    data: *const c_void,
    notification: u32,
) -> HRESULT {
    static SEEN: Mutex<u64> = Mutex::new(0);
    if command < 64 && *lock(&SEEN) & 1 << command == 0 {
        *lock(&SEEN) |= 1 << command;
        tracing::info!(command, "first coprocessor command of this kind");
    }
    // SAFETY: `this` as this function's contract gives it.
    let mut state = unsafe { state(this) };
    let word = |at: usize| -> Option<u32> {
        // SAFETY: non-null, and the command's words, per this function's
        // contract.
        (!data.is_null()).then(|| unsafe { data.cast::<u32>().add(at).read_unaligned() })
    };
    match command {
        REGISTER_MESSAGES => state.mask |= word(0).unwrap_or(0),
        UNREGISTER_MESSAGES => state.mask &= !word(0).unwrap_or(0),
        LOAD_FLOWGRAPH => {
            let flowgraph = word(1).unwrap_or(0);
            state.post(FLOWGRAPH_COMPLETED, &flowgraph.to_le_bytes());
        }
        _ if xma_command(command, word) => {}
        // shortcut: context updates and the other SHAPE units' commands are
        // acknowledged but not carried out; add them when a title sends
        // them (each kind is logged once above).
        _ => {}
    }
    if id != 0 {
        let mut done = [0u8; 16];
        done[..4].copy_from_slice(&command.to_le_bytes());
        done[4..12].copy_from_slice(&id.to_le_bytes());
        done[12..].copy_from_slice(&state.frame().to_le_bytes());
        state.post(COMMAND_COMPLETED, &done);
    }
    if notification != 0
        && let Some(word) = heap_unmap(notification)
    {
        // SAFETY: a u32 in a heap block, which the title reads as volatile.
        unsafe { word.cast::<u32>().write_volatile(1) };
    }
    S_OK
}

/// # Safety
///
/// `this` is the live `Acp` `AcpHalCreate` made. `out` is null or valid for
/// a write of a `Message`, unaligned.
unsafe extern "system" fn pop_message(this: *mut Acp, out: *mut Message) -> bool {
    // SAFETY: `this` as this function's contract gives it.
    let Some(message) = unsafe { state(this) }.messages.pop_front() else {
        return false;
    };
    if !out.is_null() {
        // SAFETY: non-null; writable, per this function's contract.
        unsafe { out.write_unaligned(message) };
    }
    true
}

/// # Safety
///
/// `this` is the live `Acp` `AcpHalCreate` made.
unsafe extern "system" fn message_count(this: *mut Acp) -> u32 {
    // SAFETY: `this` as this function's contract gives it.
    unsafe { state(this) }.messages.len() as u32
}

/// # Safety
///
/// `this` is the live `Acp` `AcpHalCreate` made, holding the only
/// reference; it must not be used after this call.
unsafe extern "system" fn release(this: *mut Acp) {
    // SAFETY: made by `AcpHalCreate` with `Box::into_raw`; released once, per
    // this function's contract.
    drop(unsafe { Box::from_raw(this) });
}

/// # Safety
///
/// `out` is null or valid for a write of a pointer.
pub unsafe extern "system" fn AcpHalCreate(out: *mut *mut Acp) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let acp = Box::new(Acp {
        vtable: &ACP_METHODS,
        state: Mutex::new(State {
            connected: false,
            mask: 0,
            messages: VecDeque::new(),
            capacity: 0,
            dropped: 0,
            since: Instant::now(),
        }),
    });
    // SAFETY: checked non-null; writable, per this function's contract.
    unsafe { out.write(Box::into_raw(acp)) };
    S_OK
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "audio",
        dlls: &["acphal.dll"],
        exports: vec![
            hle_export!(AcpHalAllocateShapeContexts),
            hle_export!(AcpHalReleaseShapeContexts),
            hle_export!(AcpHalCreate),
            hle_export!(ApuAlloc),
            hle_export!(ApuCreateHeap),
            hle_export!(ApuFree),
            hle_export!(ApuIsVirtualAddressValid),
            hle_export!(ApuMapApuAddress),
            hle_export!(ApuMapVirtualAddress),
        ],
    }
}

#[cfg(test)]
mod tests;

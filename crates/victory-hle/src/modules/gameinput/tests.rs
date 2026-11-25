use std::sync::Arc;

use super::*;
use crate::HleResolver;
use crate::HostPolicy;
use victory_loader::Provenance;
use victory_pe::ImportSymbol;

#[derive(Default)]
struct FakeHost {
    pads: [Option<PadState>; PADS],
    keys: Vec<KeyState>,
    rumble: Vec<(usize, u16, u16)>,
}

struct Fake(Arc<Mutex<FakeHost>>);

impl Source for Fake {
    fn pad(&mut self, slot: usize) -> Option<PadState> {
        self.0.lock().unwrap().pads[slot]
    }
    fn keys(&mut self) -> Vec<KeyState> {
        self.0.lock().unwrap().keys.clone()
    }
    fn rumble(&mut self, slot: usize, low: u16, high: u16) {
        self.0.lock().unwrap().rumble.push((slot, low, high));
    }
}

fn fake() -> (&'static Hub, Arc<Mutex<FakeHost>>) {
    let host = Arc::new(Mutex::new(FakeHost::default()));
    let hub = Hub::new(Box::new(Fake(Arc::clone(&host))));
    hub.poll();
    (hub, host)
}

fn key(virtual_key: u8) -> KeyState {
    KeyState { scan_code: u32::from(virtual_key) + 0x100, virtual_key, ..Default::default() }
}

fn slot<F: Copy, T>(object: *const T, offset: usize) -> F {
    // SAFETY: our objects start with their table, which has the slot.
    let f = unsafe { *(*(object as *const *const *const ())).add(offset / 8) };
    // SAFETY: the caller names the slot's signature.
    unsafe { std::mem::transmute_copy(&f) }
}

/// Device callbacks seen: device index, current and previous status.
type Log = Mutex<Vec<(usize, u32, u32)>>;

extern "system" fn record(
    _token: u64,
    context: *mut c_void,
    device: *const Device,
    _timestamp: u64,
    current: u32,
    previous: u32,
) {
    // SAFETY: the test passes its log and one of the hub's devices.
    let (log, index) = unsafe { (&*(context as *const Log), (*device).index) };
    log.lock().unwrap().push((index, current, previous));
}

fn register(hub: &Hub, kind: u32, enumeration: u32, log: &Log) -> u64 {
    type Register = extern "system" fn(
        This,
        *const Device,
        u32,
        u32,
        u32,
        *mut c_void,
        Option<DeviceCallback>,
        *mut u64,
    ) -> HRESULT;
    let mut token = 0;
    let context = log as *const Log as *mut c_void;
    let register = slot::<Register, _>(&hub.object, 0x48);
    let hr = register(
        &hub.object,
        std::ptr::null(),
        kind,
        1,
        enumeration,
        context,
        Some(record),
        &mut token,
    );
    assert_eq!(hr, S_OK);
    assert_ne!(token, 0);
    token
}

fn result(hr: HRESULT, out: *const c_void) -> Result<*const Reading, HRESULT> {
    if hr == S_OK {
        Ok(crate::test_support::valid(out).cast())
    } else {
        assert!(out.is_null());
        Err(hr)
    }
}

fn current(hub: &Hub, kind: u32, device: usize) -> Result<*const Reading, HRESULT> {
    type Current = extern "system" fn(This, u32, *const Device, Out) -> HRESULT;
    let mut out = std::ptr::null();
    let hr =
        slot::<Current, _>(&hub.object, 0x20)(&hub.object, kind, &hub.devices[device], &mut out);
    result(hr, out)
}

fn walk(hub: &Hub, offset: usize, from: *const Reading) -> Result<*const Reading, HRESULT> {
    type Walk = extern "system" fn(This, *const Reading, u32, *const Device, Out) -> HRESULT;
    let mut out = std::ptr::null();
    // SAFETY: this test's own locals, valid for the call.
    let device = unsafe { reading(from) }.device;
    let hr =
        slot::<Walk, _>(&hub.object, offset)(&hub.object, from, GAMEPAD_KIND, device, &mut out);
    result(hr, out)
}

fn pad(reading: *const Reading) -> PadState {
    let mut state = PadState::default();
    let get = slot::<extern "system" fn(*const Reading, *mut PadState) -> bool, _>(reading, 0xb0);
    assert!(get(reading, &mut state));
    state
}

fn stamp(reading: *const Reading) -> (u64, u64) {
    let time = slot::<extern "system" fn(*const Reading) -> u64, _>(reading, 0x28);
    let sequence = slot::<extern "system" fn(*const Reading, u32) -> u64, _>(reading, 0x20);
    (time(reading), sequence(reading, GAMEPAD_KIND))
}

fn release(reading: *const Reading) {
    slot::<extern "system" fn(*const Reading) -> u32, _>(reading, 0x10)(reading);
}

#[test]
fn ordinal_one_resolves_to_create() {
    let r = HleResolver::new(HostPolicy::none());
    assert_eq!(r.classify("GameInput.dll", &ImportSymbol::Ordinal(1)), Some(Provenance::Hle));
    assert_eq!(r.classify("GameInput.dll", &ImportSymbol::Ordinal(2)), None);
}

#[test]
fn pads_connect_report_and_disconnect() {
    let (hub, host) = fake();
    let log = Log::default();
    register(hub, GAMEPAD_KIND, 2, &log);
    // Blocking enumeration ran before registration returned: with no
    // host pad the keyboard's stand-in is the only one.
    assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);

    host.lock().unwrap().pads[1] = Some(PadState { buttons: A, left_x: 0.5, ..Default::default() });
    hub.poll();
    assert_eq!(
        log.lock().unwrap()[1..],
        [(1, STATUS_CONNECTED, 0), (FALLBACK, 0, STATUS_CONNECTED)]
    );

    let first = current(hub, GAMEPAD_KIND, 1).unwrap();
    assert_eq!(pad(first).buttons, A);
    assert_eq!(pad(first).left_x, 0.5);
    // No change, no new reading.
    hub.poll();
    let again = current(hub, GAMEPAD_KIND, 1).unwrap();
    assert_eq!(again, first);
    release(again);

    std::thread::sleep(Duration::from_millis(2));
    host.lock().unwrap().pads[1].as_mut().unwrap().buttons = B;
    hub.poll();
    let second = current(hub, GAMEPAD_KIND, 1).unwrap();
    assert_eq!(pad(second).buttons, B);
    let (t1, s1) = stamp(first);
    let (t2, s2) = stamp(second);
    assert!(t2 > t1 && s2 == s1 + 1);
    assert!(t2 <= now());

    // The title's edge check: previous of current, and back again.
    assert_eq!(walk(hub, 0x30, second), Ok(first));
    assert_eq!(walk(hub, 0x28, first), Ok(second));
    assert_eq!(walk(hub, 0x28, second), Err(READING_NOT_FOUND));
    assert_eq!(walk(hub, 0x30, first), Err(READING_NOT_FOUND));
    for r in [first, first, second, second] {
        release(r);
    }

    host.lock().unwrap().pads[1] = None;
    hub.poll();
    assert_eq!(
        log.lock().unwrap()[3..],
        [(1, 0, STATUS_CONNECTED), (FALLBACK, STATUS_CONNECTED, 0)]
    );
    assert_eq!(current(hub, GAMEPAD_KIND, 1), Err(DEVICE_DISCONNECTED));
    // The stand-in is back, at rest.
    let rest = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
    assert_eq!(pad(rest), PadState::default());
    release(rest);
}

#[test]
fn keys_feed_the_keyboard_and_the_stand_in_pad() {
    let (hub, host) = fake();
    let log = Log::default();
    register(hub, KEYBOARD_KIND, 2, &log);
    assert_eq!(*log.lock().unwrap(), [(KEYBOARD, STATUS_CONNECTED, 0)]);

    let keys = vec![key(b'W'), key(0x0D), key(b'L'), key(b'C')];
    host.lock().unwrap().keys = keys.clone();
    hub.poll();
    let r = current(hub, KEYBOARD_KIND, KEYBOARD).unwrap();
    let count = slot::<extern "system" fn(*const Reading) -> u32, _>(r, 0x70);
    let state = slot::<extern "system" fn(*const Reading, u32, *mut KeyState) -> u32, _>(r, 0x78);
    assert_eq!(count(r), 4);
    let mut got = [KeyState::default(); 8];
    assert_eq!(state(r, 8, got.as_mut_ptr()), 4);
    assert_eq!(got[..4], keys[..]);
    release(r);

    let r = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
    assert_eq!(
        pad(r),
        PadState {
            buttons: A,
            left_y: 1.0,
            right_x: 1.0,
            right_trigger: 1.0,
            ..Default::default()
        }
    );
    release(r);
    // A keyboard reading is no gamepad reading.
    assert_eq!(current(hub, GAMEPAD_KIND, KEYBOARD), Err(READING_NOT_FOUND));
}

#[test]
fn async_enumeration_waits_for_the_poll_and_unregistering_stops_callbacks() {
    let (hub, host) = fake();
    let log = Log::default();
    let token = register(hub, GAMEPAD_KIND, ASYNC_ENUMERATION, &log);
    assert!(log.lock().unwrap().is_empty());
    hub.poll();
    assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);

    let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(&hub.object, 0x68);
    assert!(unregister(&hub.object, token, 5000));
    host.lock().unwrap().pads[0] = Some(PadState::default());
    hub.poll();
    assert_eq!(log.lock().unwrap().len(), 1);
}

#[test]
fn unregistering_waits_for_a_running_callback() {
    use std::sync::atomic::AtomicBool;
    static INSIDE: AtomicBool = AtomicBool::new(false);
    static LEAVE: AtomicBool = AtomicBool::new(false);
    static FROM_INSIDE: AtomicU32 = AtomicU32::new(0);
    extern "system" fn slow(
        token: u64,
        context: *mut c_void,
        _: *const Device,
        _: u64,
        _: u32,
        _: u32,
    ) {
        // From inside its own callback, unregistering fails at once.
        let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(context, 0x68);
        if !unregister(context.cast(), token, 0) {
            FROM_INSIDE.fetch_add(1, Ordering::SeqCst);
        }
        INSIDE.store(true, Ordering::SeqCst);
        while !LEAVE.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    let (hub, host) = fake();
    type Register = extern "system" fn(
        This,
        *const Device,
        u32,
        u32,
        u32,
        *mut c_void,
        Option<DeviceCallback>,
        *mut u64,
    ) -> HRESULT;
    let register = slot::<Register, _>(&hub.object, 0x48);
    let mut token = 0;
    let context = (&raw const hub.object).cast_mut().cast();
    assert_eq!(
        register(
            &hub.object,
            std::ptr::null(),
            GAMEPAD_KIND,
            1,
            0,
            context,
            Some(slow),
            &mut token
        ),
        S_OK
    );
    host.lock().unwrap().pads[1] = Some(PadState::default());
    let poller = std::thread::spawn(|| hub.poll());
    while !INSIDE.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(1));
    }
    let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(&hub.object, 0x68);
    assert!(!unregister(&hub.object, token, 5000), "times out while it runs");
    let leave = std::thread::spawn(|| {
        std::thread::sleep(Duration::from_millis(50));
        LEAVE.store(true, Ordering::SeqCst);
    });
    let start = Instant::now();
    assert!(unregister(&hub.object, token, u64::MAX), "waits for it");
    assert!(start.elapsed() >= Duration::from_millis(40));
    assert_eq!(FROM_INSIDE.load(Ordering::SeqCst), 1);
    poller.join().unwrap();
    leave.join().unwrap();
}

#[test]
fn devices_are_found_by_id_and_rumble_reaches_the_host() {
    let (hub, host) = fake();
    host.lock().unwrap().pads[2] = Some(PadState::default());
    hub.poll();
    let device = &hub.devices[2] as *const Device;
    let info = slot::<extern "system" fn(*const Device) -> *const u8, _>(device, 0x18);
    // SAFETY: the info has the id at +0x20.
    let id = unsafe { *(info(device).add(0x20) as *const [u8; 32]) };
    assert_ne!(id, [0; 32]);

    type Find = extern "system" fn(This, *const [u8; 32], Out) -> HRESULT;
    let find = slot::<Find, _>(&hub.object, 0x80);
    let mut out = std::ptr::null();
    assert_eq!(find(&hub.object, &id, &mut out), S_OK);
    assert_eq!(out, device.cast());
    assert_eq!(find(&hub.object, &[1; 32], &mut out), DEVICE_NOT_FOUND);
    assert!(out.is_null());

    type Rumble = extern "system" fn(*const Device, *const [f32; 4]);
    let rumble = slot::<Rumble, _>(device, 0x50);
    rumble(device, &[0.5, 1.0, 1.0, 1.0]);
    rumble(&hub.devices[FALLBACK], &[1.0; 4]);
    rumble(device, std::ptr::null());
    assert_eq!(host.lock().unwrap().rumble, [(2, 32767, 65535), (2, 0, 0)]);
}

//! A console without a network cable: the title reaches loopback only.
//!
//! Every way out of Winsock and WinHTTP a title can import or look up by
//! name is vetted here; the rest of both DLLs is the host's own.
//!
//! * Connects (also by name, by list, `ConnectEx` from `WSAIoctl`) and
//!   datagram sends to anything but 127.0.0.0/8 and ::1 fail with
//!   `WSAENETUNREACH`.
//! * Forward lookups other than `localhost` and IP literals fail with
//!   `WSAHOST_NOT_FOUND`; reverse lookups of other addresses come back
//!   numeric, or fail if a name is required, as with no network.
//! * `WinHttpConnect` to a host that is not loopback fails with
//!   `ERROR_WINHTTP_CANNOT_CONNECT`, so no request (or WebSocket) handle can
//!   exist for one. Proxies are off and proxy auto-detection (WPAD, PAC
//!   downloads) fails with `ERROR_WINHTTP_AUTODETECTION_FAILED`.
//! * A bind to all interfaces is narrowed to loopback, so nothing on the
//!   network can connect in.
#![allow(
    non_snake_case,
    reason = "exports carry the exact API names (`WSAConnect`) that `hle_export!` binds imports by"
)]
#![allow(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "title code calls these exports through its import table or our objects' tables with pointers the real API trusts; debt tracked in AGENTS.md (lint ratchet)"
)]
#![allow(clippy::too_many_arguments, reason = "the signature is fixed by the API title code calls")]

use std::ffi::{CStr, c_char, c_void};
use std::net::{IpAddr, SocketAddr};
use std::sync::Mutex;

use windows_sys::Win32::Foundation::SetLastError;

use crate::modules::files::try_host;
use crate::{HleExport, HleModule, hle_export};

const SOCKET_ERROR: i32 = -1;
const INVALID_SOCKET: Socket = !0;
const WSAEINVAL: i32 = 10022;
const WSAEADDRNOTAVAIL: i32 = 10049;
const WSAENETDOWN: i32 = 10050;
const WSAENETUNREACH: i32 = 10051;
const WSASERVICE_NOT_FOUND: i32 = 10108;
const WSAHOST_NOT_FOUND: i32 = 11001;
const WSANO_RECOVERY: i32 = 11003;
const ERROR_WINHTTP_INTERNAL_ERROR: u32 = 12004;
const ERROR_WINHTTP_CANNOT_CONNECT: u32 = 12029;
const ERROR_WINHTTP_AUTODETECTION_FAILED: u32 = 12180;
const AF_INET: u16 = 2;
const AF_INET6: u16 = 23;
const NI_NUMERICHOST: i32 = 0x02;
const NI_NAMEREQD: i32 = 0x04;
const SIO_GET_EXTENSION_FUNCTION_POINTER: u32 = 0xC800_0006;
const SIO_GET_MULTIPLE_EXTENSION_FUNCTION_POINTER: u32 = 0xC800_0024;
const WINHTTP_ACCESS_TYPE_NO_PROXY: u32 = 1;
const WINHTTP_OPTION_PROXY: u32 = 38;

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct Guid(u32, u16, u16, [u8; 8]);

const WSAID_CONNECTEX: Guid =
    Guid(0x25a2_07b9, 0xddf3, 0x4660, [0x8e, 0xe9, 0x76, 0xe5, 0x8c, 0x74, 0x06, 0x3e]);
const WSAID_WSASENDMSG: Guid =
    Guid(0xa441_e712, 0x754f, 0x43ca, [0x84, 0xa7, 0x0d, 0xee, 0x44, 0xcf, 0x60, 0x6d]);

type Socket = usize;
type Handle = *mut c_void;

/// Logs `what` for `destination` the first time (no payload, ever).
fn once(what: &'static str, destination: String) {
    static SEEN: Mutex<Vec<(&str, String)>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.iter().any(|(w, d)| *w == what && *d == destination) {
        tracing::info!(%destination, "{what}");
        seen.push((what, destination));
    }
}

/// Sets the thread's Winsock error. Without the host's `WSASetLastError`
/// the code goes to `SetLastError`: `WSAGetLastError` reads the thread's
/// last-error value, the one `SetLastError` sets.
fn wsa_error(code: i32) {
    match try_host!("ws2_32.dll", "WSASetLastError": extern "system" fn(i32)) {
        Some(set) => set(code),
        None => {
            // SAFETY: no pointers.
            unsafe { SetLastError(code.cast_unsigned()) };
        }
    }
}

/// A Winsock call's failure when the host lacks the function behind it:
/// `WSAENETDOWN` (the network subsystem has failed, an error every Winsock
/// caller handles) as the thread's Winsock error, then `value`.
fn winsock_failure<T>(value: T) -> T {
    wsa_error(WSAENETDOWN);
    value
}

/// A name lookup's failure when the host lacks the function behind it:
/// `WSANO_RECOVERY` (`EAI_FAIL`, a nonrecoverable failure), set as the
/// thread's Winsock error and returned, as the lookup functions do.
fn lookup_failure() -> i32 {
    wsa_error(WSANO_RECOVERY);
    WSANO_RECOVERY
}

/// A WinHTTP call's failure when the host lacks the function behind it:
/// `ERROR_WINHTTP_INTERNAL_ERROR` as the thread's last error, then `value`.
fn winhttp_failure<T>(value: T) -> T {
    // SAFETY: no pointers.
    unsafe { SetLastError(ERROR_WINHTTP_INTERNAL_ERROR) };
    value
}

fn narrow(text: *const c_char) -> Option<String> {
    (!text.is_null()).then(|| {
        // SAFETY: the title passes a NUL-terminated string.
        unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned()
    })
}

fn wide(text: *const u16) -> Option<String> {
    // SAFETY: the title passes a NUL-terminated string or null.
    (!text.is_null()).then(|| unsafe {
        let len = (0..).take_while(|&i| *text.add(i) != 0).count();
        String::from_utf16_lossy(std::slice::from_raw_parts(text, len))
    })
}

/// An IPv4 or IPv6 `sockaddr` of `len` bytes.
fn socket_addr(addr: *const u8, len: i32) -> Option<SocketAddr> {
    let len = usize::try_from(len).ok()?;
    if addr.is_null() || len < 4 {
        return None;
    }
    // SAFETY: the title passes `len` readable bytes.
    let b = unsafe { std::slice::from_raw_parts(addr, len) };
    let port = u16::from_be_bytes([b[2], b[3]]);
    match u16::from_le_bytes([b[0], b[1]]) {
        AF_INET if len >= 16 => Some((<[u8; 4]>::try_from(&b[4..8]).ok()?, port).into()),
        AF_INET6 if len >= 28 => Some((<[u8; 16]>::try_from(&b[8..24]).ok()?, port).into()),
        _ => None,
    }
}

/// A raw address of family `kind` (`gethostbyaddr` style).
fn raw_ip(addr: *const c_char, len: i32, kind: i32) -> Option<IpAddr> {
    let len = usize::try_from(len).ok()?;
    if addr.is_null() {
        return None;
    }
    // SAFETY: the title passes `len` readable bytes.
    let b = unsafe { std::slice::from_raw_parts(addr.cast::<u8>(), len) };
    match (u16::try_from(kind).ok()?, len) {
        (AF_INET, 4..) => Some(<[u8; 4]>::try_from(&b[..4]).ok()?.into()),
        (AF_INET6, 16..) => Some(<[u8; 16]>::try_from(&b[..16]).ok()?.into()),
        _ => None,
    }
}

fn loopback(ip: IpAddr) -> bool {
    ip.to_canonical().is_loopback()
}

/// `localhost` or a loopback literal (bracketed IPv6 too).
fn loopback_name(name: &str) -> bool {
    let bare = name.trim_start_matches('[').trim_end_matches(']');
    bare.eq_ignore_ascii_case("localhost") || bare.parse().is_ok_and(loopback)
}

fn shown(addr: Option<impl ToString>) -> String {
    addr.map_or_else(|| "not an IP address".into(), |a| a.to_string())
}

/// True for a loopback `sockaddr`; otherwise logs it and sets
/// `WSAENETUNREACH`. Other address families count as off the machine.
fn reachable(what: &'static str, addr: *const u8, len: i32) -> bool {
    match socket_addr(addr, len) {
        Some(a) if loopback(a.ip()) => true,
        a => {
            once(what, shown(a));
            wsa_error(WSAENETUNREACH);
            false
        }
    }
}

/// Connecting by name: only to loopback. A name would need a lookup
/// (`WSAHOST_NOT_FOUND`), an address a route (`WSAENETUNREACH`).
fn reachable_name(name: Option<String>) -> bool {
    match name {
        Some(name) if !loopback_name(&name) => {
            let literal = name.parse::<IpAddr>().is_ok();
            once("network connect blocked", name);
            wsa_error(if literal { WSAENETUNREACH } else { WSAHOST_NOT_FOUND });
            false
        }
        // No name: the host rejects the call itself.
        _ => true,
    }
}

/// Names answered without asking the network: none, `localhost`, IP
/// literals (connecting to one is vetted separately).
fn resolvable(name: Option<String>) -> bool {
    match name {
        Some(name)
            if !(name.is_empty()
                || name.eq_ignore_ascii_case("localhost")
                || name.parse::<IpAddr>().is_ok()) =>
        {
            once("network name lookup blocked", name);
            wsa_error(WSAHOST_NOT_FOUND);
            false
        }
        _ => true,
    }
}

/// Reverse lookups ask DNS unless the address is loopback.
fn reverse_resolvable(ip: Option<IpAddr>) -> bool {
    match ip {
        Some(ip) if loopback(ip) => true,
        ip => {
            once("network reverse lookup blocked", shown(ip));
            false
        }
    }
}

/// `getnameinfo` flags that keep a host name lookup on the machine: the
/// numeric form, as without a network, or `None` if a name is required.
fn name_info_flags(sa: *const u8, len: i32, host_wanted: bool, flags: i32) -> Option<i32> {
    if !host_wanted
        || flags & NI_NUMERICHOST != 0
        || reverse_resolvable(socket_addr(sa, len).map(|a| a.ip()))
    {
        Some(flags)
    } else if flags & NI_NAMEREQD != 0 {
        wsa_error(WSAHOST_NOT_FOUND);
        None
    } else {
        Some(flags | NI_NUMERICHOST)
    }
}

// Connects.

pub extern "system" fn connect(s: Socket, name: *const u8, len: i32) -> i32 {
    if !reachable("network connect blocked", name, len) {
        return SOCKET_ERROR;
    }
    match try_host!("ws2_32.dll", "connect": extern "system" fn(Socket, *const u8, i32) -> i32) {
        Some(f) => f(s, name, len),
        None => winsock_failure(SOCKET_ERROR),
    }
}

pub extern "system" fn WSAConnect(
    s: Socket,
    name: *const u8,
    len: i32,
    caller_data: *const c_void,
    callee_data: *mut c_void,
    sqos: *const c_void,
    gqos: *const c_void,
) -> i32 {
    if !reachable("network connect blocked", name, len) {
        return SOCKET_ERROR;
    }
    match try_host!("ws2_32.dll", "WSAConnect": extern "system" fn(Socket, *const u8, i32, *const c_void, *mut c_void, *const c_void, *const c_void) -> i32)
    {
        Some(f) => f(s, name, len, caller_data, callee_data, sqos, gqos),
        None => winsock_failure(SOCKET_ERROR),
    }
}

pub extern "system" fn WSAConnectByNameA(
    s: Socket,
    node: *const c_char,
    service: *const c_char,
    local_len: *mut u32,
    local: *mut c_void,
    remote_len: *mut u32,
    remote: *mut c_void,
    timeout: *const c_void,
    reserved: *mut c_void,
) -> i32 {
    if !reachable_name(narrow(node)) {
        return 0;
    }
    match try_host!("ws2_32.dll", "WSAConnectByNameA": extern "system" fn(Socket, *const c_char, *const c_char, *mut u32, *mut c_void, *mut u32, *mut c_void, *const c_void, *mut c_void) -> i32)
    {
        Some(f) => f(
            s, node, service, local_len, local, remote_len, remote, timeout, reserved,
        ),
        None => winsock_failure(0),
    }
}

pub extern "system" fn WSAConnectByNameW(
    s: Socket,
    node: *const u16,
    service: *const u16,
    local_len: *mut u32,
    local: *mut c_void,
    remote_len: *mut u32,
    remote: *mut c_void,
    timeout: *const c_void,
    reserved: *mut c_void,
) -> i32 {
    if !reachable_name(wide(node)) {
        return 0;
    }
    match try_host!("ws2_32.dll", "WSAConnectByNameW": extern "system" fn(Socket, *const u16, *const u16, *mut u32, *mut c_void, *mut u32, *mut c_void, *const c_void, *mut c_void) -> i32)
    {
        Some(f) => f(
            s, node, service, local_len, local, remote_len, remote, timeout, reserved,
        ),
        None => winsock_failure(0),
    }
}

/// `SOCKET_ADDRESS`; a `SOCKET_ADDRESS_LIST` is an `i32` count, then these
/// (pointer aligned).
#[repr(C)]
struct SocketAddress {
    addr: *const u8,
    len: i32,
}

/// Every candidate must be loopback.
pub extern "system" fn WSAConnectByList(
    s: Socket,
    list: *const u8,
    local_len: *mut u32,
    local: *mut c_void,
    remote_len: *mut u32,
    remote: *mut c_void,
    timeout: *const c_void,
    reserved: *mut c_void,
) -> i32 {
    if !list.is_null() {
        // SAFETY: the title passes a `SOCKET_ADDRESS_LIST`.
        let entries = unsafe {
            let count = usize::try_from(list.cast::<i32>().read()).unwrap_or(0);
            std::slice::from_raw_parts(list.add(8).cast::<SocketAddress>(), count)
        };
        if !entries.iter().all(|e| reachable("network connect blocked", e.addr, e.len)) {
            return 0;
        }
    }
    match try_host!("ws2_32.dll", "WSAConnectByList": extern "system" fn(Socket, *const u8, *mut u32, *mut c_void, *mut u32, *mut c_void, *const c_void, *mut c_void) -> i32)
    {
        Some(f) => f(
            s, list, local_len, local, remote_len, remote, timeout, reserved,
        ),
        None => winsock_failure(0),
    }
}

pub extern "system" fn WSAJoinLeaf(
    s: Socket,
    name: *const u8,
    len: i32,
    caller_data: *const c_void,
    callee_data: *mut c_void,
    sqos: *const c_void,
    gqos: *const c_void,
    flags: u32,
) -> Socket {
    if !reachable("network connect blocked", name, len) {
        return INVALID_SOCKET;
    }
    match try_host!("ws2_32.dll", "WSAJoinLeaf": extern "system" fn(Socket, *const u8, i32, *const c_void, *mut c_void, *const c_void, *const c_void, u32) -> Socket)
    {
        Some(f) => f(s, name, len, caller_data, callee_data, sqos, gqos, flags),
        None => winsock_failure(INVALID_SOCKET),
    }
}

/// `ConnectEx` as `WSAIoctl` hands it out: vetted, then the socket's own.
extern "system" fn connect_ex(
    s: Socket,
    name: *const u8,
    len: i32,
    send: *const c_void,
    send_len: u32,
    sent: *mut u32,
    overlapped: *mut c_void,
) -> i32 {
    if !reachable("network connect blocked", name, len) {
        return 0;
    }
    let (mut real, mut size) = (0usize, 0u32);
    let Some(ioctl) = try_host!("ws2_32.dll", "WSAIoctl": extern "system" fn(Socket, u32, *const c_void, u32, *mut c_void, u32, *mut u32, *mut c_void, *const c_void) -> i32)
    else {
        return winsock_failure(0);
    };
    let guid = WSAID_CONNECTEX;
    let found = ioctl(
        s,
        SIO_GET_EXTENSION_FUNCTION_POINTER,
        (&raw const guid).cast(),
        16,
        (&raw mut real).cast(),
        8,
        &mut size,
        std::ptr::null_mut(),
        std::ptr::null(),
    );
    if found != 0 || real == 0 {
        return 0;
    }
    // SAFETY: the provider's `ConnectEx` for this socket.
    let real = unsafe {
        std::mem::transmute::<
            usize,
            extern "system" fn(
                Socket,
                *const u8,
                i32,
                *const c_void,
                u32,
                *mut u32,
                *mut c_void,
            ) -> i32,
        >(real)
    };
    real(s, name, len, send, send_len, sent, overlapped)
}

/// The `ConnectEx` and `WSASendMsg` pointers it hands out are ours; the
/// multi-pointer query (Registered I/O, whose `RIOSendEx` takes any
/// address) is refused as by an old system. Other controls pass through.
pub extern "system" fn WSAIoctl(
    s: Socket,
    code: u32,
    input: *const c_void,
    input_len: u32,
    output: *mut c_void,
    output_len: u32,
    returned: *mut u32,
    overlapped: *mut c_void,
    completion: *const c_void,
) -> i32 {
    if code == SIO_GET_MULTIPLE_EXTENSION_FUNCTION_POINTER {
        wsa_error(WSAEINVAL);
        return SOCKET_ERROR;
    }
    let Some(ioctl) = try_host!("ws2_32.dll", "WSAIoctl": extern "system" fn(Socket, u32, *const c_void, u32, *mut c_void, u32, *mut u32, *mut c_void, *const c_void) -> i32)
    else {
        return winsock_failure(SOCKET_ERROR);
    };
    let result = ioctl(
        s, code, input, input_len, output, output_len, returned, overlapped, completion,
    );
    if result == 0 && code == SIO_GET_EXTENSION_FUNCTION_POINTER && input_len >= 16 {
        // SAFETY: the host just read a GUID from `input` and wrote a pointer to `output`.
        unsafe {
            let ours = match input.cast::<Guid>().read_unaligned() {
                WSAID_CONNECTEX => connect_ex as *const () as usize,
                WSAID_WSASENDMSG => WSASendMsg as *const () as usize,
                _ => return result,
            };
            output.cast::<usize>().write_unaligned(ours);
        }
    }
    result
}

// Datagram sends.

/// Without an address it sends to the connected peer, which `connect` vetted;
/// so does a stream socket whatever `to` says (Winsock ignores it there).
pub extern "system" fn sendto(
    s: Socket,
    buf: *const u8,
    len: i32,
    flags: i32,
    to: *const u8,
    to_len: i32,
) -> i32 {
    if !to.is_null() && !stream(s) && !reachable("network datagram blocked", to, to_len) {
        return SOCKET_ERROR;
    }
    match try_host!("ws2_32.dll", "sendto": extern "system" fn(Socket, *const u8, i32, i32, *const u8, i32) -> i32)
    {
        Some(f) => f(s, buf, len, flags, to, to_len),
        None => winsock_failure(SOCKET_ERROR),
    }
}

pub extern "system" fn WSASendTo(
    s: Socket,
    buffers: *const c_void,
    count: u32,
    sent: *mut u32,
    flags: u32,
    to: *const u8,
    to_len: i32,
    overlapped: *mut c_void,
    completion: *const c_void,
) -> i32 {
    if !to.is_null() && !stream(s) && !reachable("network datagram blocked", to, to_len) {
        return SOCKET_ERROR;
    }
    match try_host!("ws2_32.dll", "WSASendTo": extern "system" fn(Socket, *const c_void, u32, *mut u32, u32, *const u8, i32, *mut c_void, *const c_void) -> i32)
    {
        Some(f) => f(
            s, buffers, count, sent, flags, to, to_len, overlapped, completion,
        ),
        None => winsock_failure(SOCKET_ERROR),
    }
}

/// A connection-oriented socket, whose sends go to the peer `connect` vetted.
/// Without the host's `getsockopt` no socket counts as one, so every
/// destination is still vetted.
fn stream(s: Socket) -> bool {
    const SOL_SOCKET: i32 = 0xffff;
    const SO_TYPE: i32 = 0x1008;
    const SOCK_STREAM: i32 = 1;
    let (mut kind, mut len) = (0i32, 4i32);
    let Some(getsockopt) = try_host!("ws2_32.dll", "getsockopt": extern "system" fn(Socket, i32, i32, *mut i32, *mut i32) -> i32)
    else {
        return false;
    };
    getsockopt(s, SOL_SOCKET, SO_TYPE, &mut kind, &mut len) == 0 && kind == SOCK_STREAM
}

/// The start of a `WSAMSG`: the destination.
#[repr(C)]
struct MessageName {
    name: *const u8,
    len: i32,
}

pub extern "system" fn WSASendMsg(
    s: Socket,
    message: *const c_void,
    flags: u32,
    sent: *mut u32,
    overlapped: *mut c_void,
    completion: *const c_void,
) -> i32 {
    // SAFETY: the title passes a `WSAMSG` or null.
    if let Some(m) = unsafe { message.cast::<MessageName>().as_ref() }
        && !m.name.is_null()
        && !reachable("network datagram blocked", m.name, m.len)
    {
        return SOCKET_ERROR;
    }
    match try_host!("ws2_32.dll", "WSASendMsg": extern "system" fn(Socket, *const c_void, u32, *mut u32, *mut c_void, *const c_void) -> i32)
    {
        Some(f) => f(s, message, flags, sent, overlapped, completion),
        None => winsock_failure(SOCKET_ERROR),
    }
}

/// All interfaces become loopback, the only one an unplugged console has;
/// any other address is not ours (`WSAEADDRNOTAVAIL`).
// shortcut: `[::]` becomes `[::1]`, so a dual-stack listener no longer hears
// 127.0.0.1; bind both families if a title relies on that.
pub extern "system" fn bind(s: Socket, name: *const u8, len: i32) -> i32 {
    let Some(host_bind) =
        try_host!("ws2_32.dll", "bind": extern "system" fn(Socket, *const u8, i32) -> i32)
    else {
        return winsock_failure(SOCKET_ERROR);
    };
    match socket_addr(name, len) {
        Some(a) if loopback(a.ip()) => host_bind(s, name, len),
        Some(a) if a.ip().is_unspecified() => {
            once("network bind narrowed to loopback", a.to_string());
            // SAFETY: `socket_addr` checked `len` readable bytes.
            let mut local = unsafe { std::slice::from_raw_parts(name, len as usize) }.to_vec();
            match a {
                SocketAddr::V4(_) => local[4..8].copy_from_slice(&[127, 0, 0, 1]),
                SocketAddr::V6(_) => local[23] = 1,
            }
            host_bind(s, local.as_ptr(), len)
        }
        a => {
            once("network bind blocked", shown(a));
            wsa_error(WSAEADDRNOTAVAIL);
            SOCKET_ERROR
        }
    }
}

// Forward lookups.

pub extern "system" fn getaddrinfo(
    node: *const c_char,
    service: *const c_char,
    hints: *const c_void,
    result: *mut *mut c_void,
) -> i32 {
    if !resolvable(narrow(node)) {
        return WSAHOST_NOT_FOUND;
    }
    match try_host!("ws2_32.dll", "getaddrinfo": extern "system" fn(*const c_char, *const c_char, *const c_void, *mut *mut c_void) -> i32)
    {
        Some(f) => f(node, service, hints, result),
        None => lookup_failure(),
    }
}

pub extern "system" fn GetAddrInfoW(
    node: *const u16,
    service: *const u16,
    hints: *const c_void,
    result: *mut *mut c_void,
) -> i32 {
    if !resolvable(wide(node)) {
        return WSAHOST_NOT_FOUND;
    }
    match try_host!("ws2_32.dll", "GetAddrInfoW": extern "system" fn(*const u16, *const u16, *const c_void, *mut *mut c_void) -> i32)
    {
        Some(f) => f(node, service, hints, result),
        None => lookup_failure(),
    }
}

pub extern "system" fn GetAddrInfoExA(
    name: *const c_char,
    service: *const c_char,
    namespace: u32,
    provider: *const c_void,
    hints: *const c_void,
    result: *mut *mut c_void,
    timeout: *const c_void,
    overlapped: *mut c_void,
    completion: *const c_void,
    handle: *mut Handle,
) -> i32 {
    if !resolvable(narrow(name)) {
        return WSAHOST_NOT_FOUND;
    }
    match try_host!("ws2_32.dll", "GetAddrInfoExA": extern "system" fn(*const c_char, *const c_char, u32, *const c_void, *const c_void, *mut *mut c_void, *const c_void, *mut c_void, *const c_void, *mut Handle) -> i32)
    {
        Some(f) => f(
            name, service, namespace, provider, hints, result, timeout, overlapped, completion,
            handle,
        ),
        None => lookup_failure(),
    }
}

pub extern "system" fn GetAddrInfoExW(
    name: *const u16,
    service: *const u16,
    namespace: u32,
    provider: *const c_void,
    hints: *const c_void,
    result: *mut *mut c_void,
    timeout: *const c_void,
    overlapped: *mut c_void,
    completion: *const c_void,
    handle: *mut Handle,
) -> i32 {
    if !resolvable(wide(name)) {
        return WSAHOST_NOT_FOUND;
    }
    match try_host!("ws2_32.dll", "GetAddrInfoExW": extern "system" fn(*const u16, *const u16, u32, *const c_void, *const c_void, *mut *mut c_void, *const c_void, *mut c_void, *const c_void, *mut Handle) -> i32)
    {
        Some(f) => f(
            name, service, namespace, provider, hints, result, timeout, overlapped, completion,
            handle,
        ),
        None => lookup_failure(),
    }
}

pub extern "system" fn gethostbyname(name: *const c_char) -> *mut c_void {
    if !resolvable(narrow(name)) {
        return std::ptr::null_mut();
    }
    match try_host!("ws2_32.dll", "gethostbyname": extern "system" fn(*const c_char) -> *mut c_void)
    {
        Some(f) => f(name),
        None => winsock_failure(std::ptr::null_mut()),
    }
}

pub extern "system" fn WSAAsyncGetHostByName(
    window: Handle,
    message: u32,
    name: *const c_char,
    buf: *mut c_char,
    buf_len: i32,
) -> Handle {
    if !resolvable(narrow(name)) {
        return std::ptr::null_mut();
    }
    match try_host!("ws2_32.dll", "WSAAsyncGetHostByName": extern "system" fn(Handle, u32, *const c_char, *mut c_char, i32) -> Handle)
    {
        Some(f) => f(window, message, name, buf, buf_len),
        None => winsock_failure(std::ptr::null_mut()),
    }
}

/// Name-space queries (DNS, also network lists and Bluetooth) are refused
/// as a console without a network answers them.
// shortcut: local name spaces are refused too; pass those through if a title
// needs one.
pub extern "system" fn WSALookupServiceBeginA(
    _restrictions: *const c_void,
    _flags: u32,
    _lookup: *mut Handle,
) -> i32 {
    once("network name-space query blocked", "any".into());
    wsa_error(WSASERVICE_NOT_FOUND);
    SOCKET_ERROR
}

/// See [`WSALookupServiceBeginA`].
pub extern "system" fn WSALookupServiceBeginW(
    restrictions: *const c_void,
    flags: u32,
    lookup: *mut Handle,
) -> i32 {
    WSALookupServiceBeginA(restrictions, flags, lookup)
}

// Reverse lookups.

pub extern "system" fn gethostbyaddr(addr: *const c_char, len: i32, kind: i32) -> *mut c_void {
    if !reverse_resolvable(raw_ip(addr, len, kind)) {
        wsa_error(WSAHOST_NOT_FOUND);
        return std::ptr::null_mut();
    }
    match try_host!("ws2_32.dll", "gethostbyaddr": extern "system" fn(*const c_char, i32, i32) -> *mut c_void)
    {
        Some(f) => f(addr, len, kind),
        None => winsock_failure(std::ptr::null_mut()),
    }
}

pub extern "system" fn WSAAsyncGetHostByAddr(
    window: Handle,
    message: u32,
    addr: *const c_char,
    len: i32,
    kind: i32,
    buf: *mut c_char,
    buf_len: i32,
) -> Handle {
    if !reverse_resolvable(raw_ip(addr, len, kind)) {
        wsa_error(WSAHOST_NOT_FOUND);
        return std::ptr::null_mut();
    }
    match try_host!("ws2_32.dll", "WSAAsyncGetHostByAddr": extern "system" fn(Handle, u32, *const c_char, i32, i32, *mut c_char, i32) -> Handle)
    {
        Some(f) => f(window, message, addr, len, kind, buf, buf_len),
        None => winsock_failure(std::ptr::null_mut()),
    }
}

pub extern "system" fn getnameinfo(
    sa: *const u8,
    sa_len: i32,
    host: *mut c_char,
    host_len: u32,
    serv: *mut c_char,
    serv_len: u32,
    flags: i32,
) -> i32 {
    let Some(flags) = name_info_flags(sa, sa_len, !host.is_null(), flags) else {
        return WSAHOST_NOT_FOUND;
    };
    match try_host!("ws2_32.dll", "getnameinfo": extern "system" fn(*const u8, i32, *mut c_char, u32, *mut c_char, u32, i32) -> i32)
    {
        Some(f) => f(sa, sa_len, host, host_len, serv, serv_len, flags),
        None => lookup_failure(),
    }
}

pub extern "system" fn GetNameInfoW(
    sa: *const u8,
    sa_len: i32,
    host: *mut u16,
    host_len: u32,
    serv: *mut u16,
    serv_len: u32,
    flags: i32,
) -> i32 {
    let Some(flags) = name_info_flags(sa, sa_len, !host.is_null(), flags) else {
        return WSAHOST_NOT_FOUND;
    };
    match try_host!("ws2_32.dll", "GetNameInfoW": extern "system" fn(*const u8, i32, *mut u16, u32, *mut u16, u32, i32) -> i32)
    {
        Some(f) => f(sa, sa_len, host, host_len, serv, serv_len, flags),
        None => lookup_failure(),
    }
}

// WinHTTP.

/// The one door to a server: requests, uploads and WebSockets all hang off
/// the handle this returns.
pub extern "system" fn WinHttpConnect(
    session: Handle,
    server: *const u16,
    port: u16,
    reserved: u32,
) -> Handle {
    if let Some(name) = wide(server)
        && !loopback_name(&name)
    {
        once("HTTP connection blocked", format!("{name}:{port}"));
        // SAFETY: no pointers.
        unsafe { SetLastError(ERROR_WINHTTP_CANNOT_CONNECT) };
        return std::ptr::null_mut();
    }
    match try_host!("winhttp.dll", "WinHttpConnect": extern "system" fn(Handle, *const u16, u16, u32) -> Handle)
    {
        Some(f) => f(session, server, port, reserved),
        None => winhttp_failure(std::ptr::null_mut()),
    }
}

/// Sessions never use a proxy: only loopback is reachable, and a proxy
/// (named, or found by auto-detection) would carry it off the machine.
pub extern "system" fn WinHttpOpen(
    agent: *const u16,
    access: u32,
    _proxy: *const u16,
    _bypass: *const u16,
    flags: u32,
) -> Handle {
    if access != WINHTTP_ACCESS_TYPE_NO_PROXY {
        once("HTTP proxy ignored", format!("access type {access}"));
    }
    match try_host!("winhttp.dll", "WinHttpOpen": extern "system" fn(*const u16, u32, *const u16, *const u16, u32) -> Handle)
    {
        Some(f) => f(
            agent,
            WINHTTP_ACCESS_TYPE_NO_PROXY,
            std::ptr::null(),
            std::ptr::null(),
            flags,
        ),
        None => winhttp_failure(std::ptr::null_mut()),
    }
}

/// See [`WinHttpOpen`]: a proxy set later is accepted and not used.
pub extern "system" fn WinHttpSetOption(
    handle: Handle,
    option: u32,
    buffer: *const c_void,
    len: u32,
) -> i32 {
    if option == WINHTTP_OPTION_PROXY {
        once("HTTP proxy ignored", "WINHTTP_OPTION_PROXY".into());
        return 1;
    }
    match try_host!("winhttp.dll", "WinHttpSetOption": extern "system" fn(Handle, u32, *const c_void, u32) -> i32)
    {
        Some(f) => f(handle, option, buffer, len),
        None => winhttp_failure(0),
    }
}

/// Proxy auto-detection asks the network (DHCP, DNS, a PAC download).
fn no_auto_proxy() -> u32 {
    once("HTTP proxy auto-detection blocked", "WPAD/PAC".into());
    ERROR_WINHTTP_AUTODETECTION_FAILED
}

pub extern "system" fn WinHttpDetectAutoProxyConfigUrl(_flags: u32, _url: *mut *mut u16) -> i32 {
    // SAFETY: no pointers.
    unsafe { SetLastError(no_auto_proxy()) };
    0
}

pub extern "system" fn WinHttpGetProxyForUrl(
    _session: Handle,
    _url: *const u16,
    _options: *const c_void,
    _info: *mut c_void,
) -> i32 {
    // SAFETY: no pointers.
    unsafe { SetLastError(no_auto_proxy()) };
    0
}

pub extern "system" fn WinHttpGetProxyForUrlEx(
    _resolver: Handle,
    _url: *const u16,
    _options: *const c_void,
    _context: usize,
) -> u32 {
    no_auto_proxy()
}

pub extern "system" fn WinHttpGetProxyForUrlEx2(
    _resolver: Handle,
    _url: *const u16,
    _options: *const c_void,
    _selection_len: u32,
    _selection: *const u8,
    _context: usize,
) -> u32 {
    no_auto_proxy()
}

pub extern "system" fn WinHttpGetProxySettingsEx(
    _resolver: Handle,
    _kind: i32,
    _params: *const c_void,
    _context: usize,
) -> u32 {
    no_auto_proxy()
}

/// Winsock imports by ordinal; the Winsock 1.1 numbers are fixed.
fn ordinal(name: &'static str, f: usize) -> HleExport {
    HleExport { name, address: f }
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "net",
        dlls: &["ws2_32.dll", "winhttp.dll"],
        exports: vec![
            hle_export!(connect),
            hle_export!(WSAConnect),
            hle_export!(WSAConnectByNameA),
            hle_export!(WSAConnectByNameW),
            hle_export!(WSAConnectByList),
            hle_export!(WSAJoinLeaf),
            hle_export!(WSAIoctl),
            hle_export!(sendto),
            hle_export!(WSASendTo),
            hle_export!(WSASendMsg),
            hle_export!(bind),
            hle_export!(getaddrinfo),
            hle_export!(GetAddrInfoW),
            hle_export!(GetAddrInfoExA),
            hle_export!(GetAddrInfoExW),
            hle_export!(gethostbyname),
            hle_export!(WSAAsyncGetHostByName),
            hle_export!(WSALookupServiceBeginA),
            hle_export!(WSALookupServiceBeginW),
            hle_export!(gethostbyaddr),
            hle_export!(WSAAsyncGetHostByAddr),
            hle_export!(getnameinfo),
            hle_export!(GetNameInfoW),
            hle_export!(WinHttpConnect),
            hle_export!(WinHttpOpen),
            hle_export!(WinHttpSetOption),
            hle_export!(WinHttpDetectAutoProxyConfigUrl),
            hle_export!(WinHttpGetProxyForUrl),
            hle_export!(WinHttpGetProxyForUrlEx),
            hle_export!(WinHttpGetProxyForUrlEx2),
            hle_export!(WinHttpGetProxySettingsEx),
            ordinal("#2", bind as *const () as usize),
            ordinal("#4", connect as *const () as usize),
            ordinal("#20", sendto as *const () as usize),
            ordinal("#51", gethostbyaddr as *const () as usize),
            ordinal("#52", gethostbyname as *const () as usize),
            ordinal("#102", WSAAsyncGetHostByAddr as *const () as usize),
            ordinal("#103", WSAAsyncGetHostByName as *const () as usize),
        ],
    }
}

#[cfg(test)]
mod tests;

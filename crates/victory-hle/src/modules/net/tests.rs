use super::*;
use victory_pe::ImportSymbol;

fn v4(ip: [u8; 4], port: u16) -> [u8; 16] {
    let mut a = [0u8; 16];
    a[..2].copy_from_slice(&AF_INET.to_le_bytes());
    a[2..4].copy_from_slice(&port.to_be_bytes());
    a[4..8].copy_from_slice(&ip);
    a
}

// TEST-NET-1 (192.0.2.0/24): never routed, and never tried here.
const AWAY: [u8; 4] = [192, 0, 2, 1];
const HOME: [u8; 4] = [127, 0, 0, 1];

/// What a call returns when the host lacks the function behind it.
#[test]
#[cfg(windows)]
fn a_missing_host_export_fails_the_call() {
    let _starts_winsock = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
    let last_error = try_host!("ws2_32.dll", "WSAGetLastError": extern "system" fn() -> i32)
        .expect("ws2_32 exports WSAGetLastError");
    assert_eq!(winsock_failure(SOCKET_ERROR), SOCKET_ERROR);
    assert_eq!(last_error(), WSAENETDOWN);
    assert_eq!(lookup_failure(), WSANO_RECOVERY);
    assert_eq!(last_error(), WSANO_RECOVERY);
    assert!(winhttp_failure(std::ptr::null_mut::<c_void>()).is_null());
    // SAFETY: no arguments.
    let error = unsafe { windows_sys::Win32::Foundation::GetLastError() };
    assert_eq!(error, ERROR_WINHTTP_INTERNAL_ERROR);
}

#[test]
fn only_loopback_is_reachable() {
    // Also starts Winsock for the raw calls below.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let socket = try_host!("ws2_32.dll", "socket": extern "system" fn(i32, i32, i32) -> Socket)
        .expect("ws2_32 exports socket");
    let close = try_host!("ws2_32.dll", "closesocket": extern "system" fn(Socket) -> i32)
        .expect("ws2_32 exports closesocket");
    let last_error = try_host!("ws2_32.dll", "WSAGetLastError": extern "system" fn() -> i32)
        .expect("ws2_32 exports WSAGetLastError");
    let null = std::ptr::null_mut::<c_void>();

    // Connects.
    let s = socket(2, 1, 6);
    assert_ne!(s, INVALID_SOCKET);
    // SAFETY: this test's own locals and strings, valid for the call.
    assert_eq!(unsafe { connect(s, v4(AWAY, 80).as_ptr(), 16) }, SOCKET_ERROR);
    assert_eq!(last_error(), WSAENETUNREACH);
    let n = null.cast();
    assert_eq!(
        // SAFETY: this test's own locals and strings, valid for the call.
        unsafe {
            WSAConnectByNameA(
                s,
                c"example.com".as_ptr(),
                c"80".as_ptr(),
                n,
                null,
                n,
                null,
                null,
                null,
            )
        },
        0
    );
    assert_eq!(last_error(), WSAHOST_NOT_FOUND);
    // SAFETY: this test's own locals and strings, valid for the call.
    assert_eq!(unsafe { connect(s, v4(HOME, port).as_ptr(), 16) }, 0);
    listener.accept().unwrap();
    // A stream socket sends to its vetted peer whatever `to` says.
    // SAFETY: this test's own locals and strings, valid for the call.
    assert_eq!(unsafe { sendto(s, b"x".as_ptr(), 1, 0, v4(AWAY, 9).as_ptr(), 16) }, 1);
    close(s);

    // `ConnectEx` from `WSAIoctl` is the vetted one.
    let s = socket(2, 1, 6);
    let mut f = 0usize;
    let mut size = 0u32;
    let guid = WSAID_CONNECTEX;
    let got = WSAIoctl(
        s,
        SIO_GET_EXTENSION_FUNCTION_POINTER,
        (&raw const guid).cast(),
        16,
        (&raw mut f).cast(),
        8,
        &mut size,
        null,
        null,
    );
    assert_eq!((got, f), (0, connect_ex as *const () as usize));
    // SAFETY: this test's own locals and strings, valid for the call.
    assert_eq!(unsafe { connect_ex(s, v4(AWAY, 80).as_ptr(), 16, null, 0, n, null) }, 0);
    assert_eq!(last_error(), WSAENETUNREACH);
    close(s);

    // Datagram sends.
    let u = socket(2, 2, 17);
    // SAFETY: this test's own locals and strings, valid for the call.
    assert_eq!(unsafe { sendto(u, b"x".as_ptr(), 1, 0, v4(AWAY, 9).as_ptr(), 16) }, SOCKET_ERROR);
    assert_eq!(last_error(), WSAENETUNREACH);
    // SAFETY: this test's own locals and strings, valid for the call.
    assert_eq!(unsafe { sendto(u, b"x".as_ptr(), 1, 0, v4(HOME, port).as_ptr(), 16) }, 1);
    close(u);

    // Forward and reverse lookups.
    let mut result = null;
    assert_eq!(
        // SAFETY: this test's own locals and strings, valid for the call.
        unsafe {
            getaddrinfo(c"example.com".as_ptr(), std::ptr::null(), std::ptr::null(), &mut result)
        },
        WSAHOST_NOT_FOUND
    );
    let name: Vec<u16> = "example.com\0".encode_utf16().collect();
    assert_eq!(
        // SAFETY: this test's own locals and strings, valid for the call.
        unsafe { GetAddrInfoW(name.as_ptr(), std::ptr::null(), std::ptr::null(), &mut result) },
        WSAHOST_NOT_FOUND
    );
    let mut text = [0 as c_char; 64];
    let away = v4(AWAY, 80);
    assert_eq!(
        // SAFETY: this test's own locals and strings, valid for the call.
        unsafe {
            getnameinfo(away.as_ptr(), 16, text.as_mut_ptr(), 64, std::ptr::null_mut(), 0, 0)
        },
        0
    );
    // SAFETY: NUL-terminated by `getnameinfo`.
    let numeric = unsafe { CStr::from_ptr(text.as_ptr()) };
    assert_eq!(numeric.to_str(), Ok("192.0.2.1"));
    assert_eq!(
        // SAFETY: this test's own locals and strings, valid for the call.
        unsafe {
            getnameinfo(
                away.as_ptr(),
                16,
                text.as_mut_ptr(),
                64,
                std::ptr::null_mut(),
                0,
                NI_NAMEREQD,
            )
        },
        WSAHOST_NOT_FOUND
    );
    // SAFETY: this test's own locals and strings, valid for the call.
    assert!(unsafe { gethostbyaddr(AWAY.as_ptr().cast(), 4, 2) }.is_null());

    // WinHTTP.
    // SAFETY: this test's own locals and strings, valid for the call.
    assert!(unsafe { WinHttpConnect(null, name.as_ptr(), 443, 0) }.is_null());
    // SAFETY: no arguments.
    let error = unsafe { windows_sys::Win32::Foundation::GetLastError() };
    assert_eq!(error, ERROR_WINHTTP_CANNOT_CONNECT);
    assert_eq!(
        WinHttpGetProxyForUrlEx(null, name.as_ptr(), std::ptr::null(), 0),
        ERROR_WINHTTP_AUTODETECTION_FAILED
    );

    // The ordinals we override name the functions they stand for.
    let m = module();
    for e in m.exports.iter().filter(|e| e.name.starts_with('#')) {
        let named =
            m.exports.iter().find(|n| n.address == e.address && !n.name.starts_with('#')).unwrap();
        let by_ordinal = ImportSymbol::Ordinal(e.name[1..].parse().unwrap());
        assert_eq!(
            crate::HostPolicy::default().resolve("ws2_32.dll", &by_ordinal),
            crate::host::function("ws2_32.dll", named.name),
            "{} is not {}",
            e.name,
            named.name
        );
    }
    // Every override names a real export of its DLL. Some exports only
    // exist on newer Windows (WinHttpGetProxySettingsEx is absent from
    // Windows 10's winhttp.dll); the HLE supplies those itself, so skip
    // them only when the host really lacks them.
    const NEWER_WINDOWS_ONLY: &[&str] = &["WinHttpGetProxySettingsEx"];
    for e in m.exports.iter().filter(|e| !e.name.starts_with('#')) {
        let dll = if e.name.starts_with("WinHttp") { "winhttp.dll" } else { "ws2_32.dll" };
        let present = crate::host::function(dll, e.name).is_some();
        if !present && NEWER_WINDOWS_ONLY.contains(&e.name) {
            continue;
        }
        assert!(present, "{dll} lacks {}", e.name);
    }
}

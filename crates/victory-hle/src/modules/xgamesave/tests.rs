use std::io::BufRead;
use std::sync::atomic::{AtomicU32, Ordering};

use super::*;
use crate::modules::xgameruntime::method;

fn block() -> AsyncBlock {
    AsyncBlock {
        queue: std::ptr::null(),
        context: std::ptr::null_mut(),
        callback: None,
        internal: [0; 4],
    }
}

static STEPS: AtomicU32 = AtomicU32::new(0);

/// In a child process of the crash test, stops for good at the step
/// `VICTORY_SAVE_STOP` names and says so.
pub(super) fn crash_point() {
    let Ok(stop) = std::env::var("VICTORY_SAVE_STOP") else {
        return;
    };
    if stop.parse() == Ok(STEPS.fetch_add(1, Ordering::SeqCst) + 1) {
        println!("STOPPED");
        let _ = std::io::stdout().flush();
        loop {
            std::thread::sleep(std::time::Duration::from_secs(60));
        }
    }
}

/// The per-user quota and the host disk fail with their own errors.
#[test]
fn a_full_disk_is_not_the_quota() {
    let mb = 1 << 20;
    assert_eq!(no_room(2 * mb, 0, 0, mb), Some(OUT_OF_LOCAL_STORAGE));
    assert_eq!(no_room(2 * mb, 0, QUOTA - mb, 100 * mb), Some(QUOTA_EXCEEDED));
    // Replacing a blob of the same size fits a full quota.
    assert_eq!(no_room(mb, mb, QUOTA, 100 * mb), None);
    assert_eq!(no_room(mb, 0, 0, mb), None);
}

fn update(root: &Path, display: &str, changes: &[(&str, Option<&str>)]) -> HRESULT {
    Update {
        name: "Slot".into(),
        dir: container_dir(root, "Slot"),
        display: Some(display.into()),
        changes: changes
            .iter()
            .map(|(n, d)| (n.to_string(), d.map(|d| d.as_bytes().to_vec())))
            .collect(),
    }
    .submit()
}

/// The container `Slot`: display name and blobs.
fn state(root: &Path) -> Option<(String, Vec<(String, String)>)> {
    let c = Provider { root: root.into() }.container("Slot");
    let display = current_named(&c.dir, "Slot").unwrap()?.display;
    let blobs = read(&c, None).unwrap();
    let text = |(n, d): (String, Vec<u8>)| (n, String::from_utf8(d).unwrap());
    Some((display, blobs.into_iter().map(text).collect()))
}

fn old_state() -> Option<(String, Vec<(String, String)>)> {
    Some(("one".into(), vec![("a".into(), "old a".into()), ("b".into(), "old b".into())]))
}

/// Run by the crash test in a child process; does nothing otherwise.
#[test]
fn crash_child() {
    let (Ok(root), Ok(op)) =
        (std::env::var("VICTORY_SAVE_CHILD"), std::env::var("VICTORY_SAVE_OP"))
    else {
        return;
    };
    let root = PathBuf::from(root);
    let hr = if op == "update" {
        update(&root, "two", &[("a", Some("new a")), ("b", None), ("c", Some("new c"))])
    } else {
        delete(&root, "Slot")
    };
    assert_eq!(hr, S_OK);
    println!("FINISHED");
}

/// Kills a process in the middle of an update (and of a deletion) at
/// every step: what is left is the whole old container or the whole
/// new one.
#[test]
fn a_kill_at_any_step_leaves_old_or_new() {
    let new =
        Some(("two".into(), vec![("a".into(), "new a".into()), ("c".into(), "new c".into())]));
    for (op, after) in [("update", new), ("delete", None)] {
        let mut seen = (0, 0);
        for stop in 1.. {
            let dir = tempfile::tempdir().unwrap();
            let root = dir.path();
            assert_eq!(update(root, "one", &[("a", Some("old a")), ("b", Some("old b"))]), S_OK);
            let mut child = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "modules::xgamesave::tests::crash_child", "--nocapture"])
                .args(["--test-threads=1"])
                .env("VICTORY_SAVE_CHILD", root)
                .env("VICTORY_SAVE_OP", op)
                .env("VICTORY_SAVE_STOP", stop.to_string())
                .stdout(std::process::Stdio::piped())
                .spawn()
                .unwrap();
            let out = std::io::BufReader::new(child.stdout.take().unwrap());
            let mut finished = false;
            for line in out.lines() {
                let line = line.unwrap();
                if line.contains("STOPPED") {
                    break;
                }
                finished |= line.contains("FINISHED");
            }
            let _ = child.kill();
            child.wait().unwrap();
            let left = state(root);
            if left == old_state() {
                seen.0 += 1;
            } else {
                assert_eq!(left, after, "{op}, killed at step {stop}");
                seen.1 += 1;
            }
            if finished {
                break;
            }
        }
        assert!(seen.0 > 0 && seen.1 > 1, "{op}: {seen:?}");
    }
}

#[test]
fn names_are_exact() {
    for name in ["Save_1", "slot/2", "a.b c%", "\u{e9}t\u{e9}"] {
        let file = file_name(name);
        assert!(!file.contains(['.', '/', ' ']), "{file}");
        assert_eq!(decode(&file).as_deref(), Some(name));
    }
    assert_eq!(decode("x.new"), None);
    for good in ["Profile/Slot_1", "a/b/Save-2.sav", "x", &"a".repeat(256)] {
        assert!(valid_container(good), "{good}");
    }
    for bad in ["a b", "", "a/", "a.", "a..b", "a.b/c", "a-b/c", &"a".repeat(257)] {
        assert!(!valid_container(bad), "{bad}");
    }

    // Containers and blobs differing only in case, reserved device
    // names, characters the host forbids, long names: all distinct.
    let dir = tempfile::tempdir().unwrap();
    let p = Provider { root: dir.path().into() };
    let long = "n".repeat(300);
    let blobs = ["A", "a", "CON", "nul.txt", "x/y:z*?\"<>|", long.as_str(), "end. "];
    for container in ["Save", "save", "SAVE/CON.x"] {
        let c = p.container(container);
        let u = Update {
            name: c.name.clone(),
            dir: c.dir.clone(),
            display: None,
            changes: blobs
                .iter()
                .map(|b| (b.to_string(), Some(format!("{container}:{b}").into_bytes())))
                .collect(),
        };
        assert_eq!(u.submit(), S_OK, "{container}");
    }
    let _saves = saves_lock();
    let names: Vec<String> = containers(dir.path()).into_iter().map(|c| c.0).collect();
    drop(_saves);
    assert_eq!(names, ["SAVE/CON.x", "Save", "save"]);
    for container in ["Save", "save", "SAVE/CON.x"] {
        let read = read(&p.container(container), None).unwrap();
        let mut expected: Vec<_> = blobs
            .iter()
            .map(|b| (b.to_string(), format!("{container}:{b}").into_bytes()))
            .collect();
        expected.sort();
        assert_eq!(read, expected, "{container}");
    }
}

#[test]
fn the_old_layout_moves_over() {
    let saves = tempfile::tempdir().unwrap();
    let root = saves.path().join(format!("{:016X}", xuser::XUID)).join("scid");
    let old = root.join("Player");
    std::fs::create_dir_all(&old).unwrap();
    std::fs::write(old.join("Save%20Game%20Data"), b"data").unwrap();
    std::fs::write(old.join("Save%20Game%20Header"), b"header").unwrap();
    std::fs::write(old.join("Save%20Game%20Data.new"), b"half").unwrap();
    std::fs::write(old.join(".display"), b"Chapter 1").unwrap();
    let p = Provider::open(saves.path(), Some("scid")).unwrap();
    assert!(!old.exists());
    let c = p.container("Player");
    let blobs = read(&c, None).unwrap();
    assert_eq!(
        blobs,
        [
            ("Save Game Data".into(), b"data".to_vec()),
            ("Save Game Header".into(), b"header".to_vec())
        ]
    );
    let v = current_named(&c.dir, "Player").unwrap().unwrap();
    assert_eq!((v.number, v.display.as_str()), (1, "Chapter 1"));
}

/// An empty configuration id names none. As a folder name it would be
/// the folder every configuration lives in, whose folders then look
/// like containers of the old layout and are converted away.
#[test]
fn an_empty_configuration_id_is_the_default_one() {
    let saves = tempfile::tempdir().unwrap();
    let scid = Provider::open(saves.path(), Some("scid")).unwrap();
    assert_eq!(update(&scid.root, "one", &[("a", Some("kept"))]), S_OK);
    let kept = Some(("one".into(), vec![("a".into(), "kept".into())]));
    assert_eq!(state(&scid.root), kept);

    let empty = Provider::open(saves.path(), Some("")).unwrap();
    assert!(scid.root.is_dir(), "another configuration's saves are gone");
    assert_eq!(state(&scid.root), kept);
    let default = Provider::open(saves.path(), None).unwrap();
    assert_eq!(empty.root, default.root);

    // Named configurations stay apart, from it and from each other.
    assert_ne!(empty.root, scid.root);
    assert_eq!(state(&empty.root), None);
    let other = Provider::open(saves.path(), Some("other")).unwrap();
    assert_eq!(state(&other.root), None);
    assert_eq!(state(&scid.root), kept);
}

#[test]
fn saves_persist_across_providers() {
    let this: This = &OBJECT;
    let saves = tempfile::tempdir().unwrap();
    let provider = give(Provider::open(saves.path(), Some("scid")).unwrap());

    // Write two blobs through an update.
    let mut container = std::ptr::null_mut();
    type Create =
        extern "system" fn(This, *const Provider, *const c_char, *mut *mut Container) -> HRESULT;
    let create = method::<Create>(&OBJECT, 0x80);
    assert_eq!(
        create(this, provider, c"bad name".as_ptr(), &mut container),
        INVALID_CONTAINER_NAME
    );
    assert_eq!(create(this, provider, c"Slot_1".as_ptr(), &mut container), S_OK);
    let mut update = std::ptr::null_mut();
    type Update_ =
        extern "system" fn(This, *const Container, *const c_char, *mut *mut Update) -> HRESULT;
    assert_eq!(
        method::<Update_>(&OBJECT, 0xb8)(this, container, c"Chapter 1".as_ptr(), &mut update),
        S_OK
    );
    type Write = extern "system" fn(This, *mut Update, *const c_char, *const u8, usize) -> HRESULT;
    let write = method::<Write>(&OBJECT, 0xc8);
    assert_eq!(write(this, update, c"world".as_ptr(), b"hello".as_ptr(), 5), S_OK);
    assert_eq!(write(this, update, c"player.dat".as_ptr(), b"xy".as_ptr(), 2), S_OK);
    assert_eq!(write(this, update, c"gone".as_ptr(), b"z".as_ptr(), 1), S_OK);
    type Delete = extern "system" fn(This, *mut Update, *const c_char) -> HRESULT;
    assert_eq!(method::<Delete>(&OBJECT, 0xd0)(this, update, c"gone".as_ptr()), S_OK);
    type Submit = extern "system" fn(This, *const Update, *mut AsyncBlock) -> HRESULT;
    let mut b = block();
    assert_eq!(method::<Submit>(&OBJECT, 0xe0)(this, update, &mut b), S_OK);
    assert_eq!(xasync::XAsyncGetStatus(&mut b, true), S_OK);
    // SAFETY: this test's own locals, valid for the call.
    unsafe { close_update(this, update) };
    // SAFETY: this test's own locals, valid for the call.
    unsafe { close_container(this, container) };
    // SAFETY: this test's own locals, valid for the call.
    unsafe { close_provider(this, provider) };

    // A new provider sees the container and reads both blobs back.
    let provider = give(Provider::open(saves.path(), Some("scid")).unwrap());
    extern "system" fn seen(info: *const ContainerInfo, context: *mut c_void) -> bool {
        // SAFETY: valid during the callback; the context is the test's vector.
        let (info, out) =
            unsafe { (&*info, &mut *context.cast::<Vec<(String, String, u32, u64)>>()) };
        out.push((
            text(info.name).unwrap(),
            text(info.display_name).unwrap(),
            info.blob_count,
            info.total_size,
        ));
        true
    }
    let mut found: Vec<(String, String, u32, u64)> = Vec::new();
    let context = (&raw mut found).cast();
    // SAFETY: this test's own locals, valid for the call.
    assert_eq!(unsafe { enumerate_containers(this, provider, context, Some(seen)) }, S_OK);
    assert_eq!(found, [("Slot_1".into(), "Chapter 1".into(), 2, 7)]);

    let mut container = std::ptr::null_mut();
    assert_eq!(
        // SAFETY: this test's own locals, valid for the call.
        unsafe { create_container(this, provider, c"Slot_1".as_ptr(), &mut container) },
        S_OK
    );
    let mut b = block();
    let wanted = [c"world".as_ptr(), c"player.dat".as_ptr()];
    // SAFETY: this test's own locals, valid for the call.
    assert_eq!(unsafe { read_async(this, container, wanted.as_ptr(), 2, &mut b) }, S_OK);
    assert_eq!(xasync::XAsyncGetStatus(&mut b, true), S_OK);
    let mut size = 0;
    assert_eq!(xasync::XAsyncGetResultSize(&mut b, &mut size), S_OK);
    assert_eq!(size, 2 * BLOB + "world\0hello".len() + "player.dat\0xy".len());
    let mut buffer = vec![0u64; size.div_ceil(8)];
    let mut count = 0;
    assert_eq!(
        // SAFETY: this test's own locals, valid for the call.
        unsafe { read_result(this, &mut b, 8, buffer.as_mut_ptr().cast(), &mut count) },
        BUFFER_TOO_SMALL
    );
    assert_eq!(
        // SAFETY: this test's own locals, valid for the call.
        unsafe { read_result(this, &mut b, size, buffer.as_mut_ptr().cast(), &mut count) },
        S_OK
    );
    assert_eq!(count, 2);
    let blob = |i: usize| {
        // SAFETY: the layout the call wrote.
        unsafe {
            let entry = buffer.as_ptr().cast::<u8>().add(i * BLOB);
            let name = text(entry.cast::<*const c_char>().read_unaligned()).unwrap();
            let len = entry.add(8).cast::<u32>().read_unaligned() as usize;
            let data =
                std::slice::from_raw_parts(entry.add(16).cast::<*const u8>().read_unaligned(), len);
            (name, data.to_vec())
        }
    };
    assert_eq!(blob(0), ("world".into(), b"hello".to_vec()));
    assert_eq!(blob(1), ("player.dat".into(), b"xy".to_vec()));

    let missing = [c"nothing".as_ptr()];
    let mut one = 1;
    let mut small = [0u8; 8];
    let read_now = method::<
        extern "system" fn(
            This,
            *const Container,
            *const *const c_char,
            *mut u32,
            usize,
            *mut u8,
        ) -> HRESULT,
    >(&OBJECT, 0xa0);
    assert_eq!(
        read_now(this, container, missing.as_ptr(), &mut one, 8, small.as_mut_ptr()),
        BLOB_NOT_FOUND
    );
    assert_eq!(
        read_now(this, container, wanted.as_ptr(), &mut 2, 8, small.as_mut_ptr()),
        BUFFER_TOO_SMALL
    );

    // Quota counts the blobs; deleting the container frees it.
    let mut b = block();
    // SAFETY: this test's own locals, valid for the call.
    assert_eq!(unsafe { quota_async(this, provider, &mut b) }, S_OK);
    let mut left = 0i64;
    // SAFETY: this test's own locals, valid for the call.
    assert_eq!(unsafe { quota_result(this, &mut b, &mut left) }, S_OK);
    assert!(left as u64 <= QUOTA - 7);
    let mut b = block();
    // SAFETY: this test's own locals, valid for the call.
    assert_eq!(unsafe { delete_async(this, provider, c"Slot_1".as_ptr(), &mut b) }, S_OK);
    // SAFETY: this test's own locals, valid for the call.
    assert_eq!(unsafe { delete_result(this, &mut b) }, S_OK);
    found.clear();
    let context = (&raw mut found).cast();
    // SAFETY: this test's own locals, valid for the call.
    assert_eq!(unsafe { enumerate_containers(this, provider, context, Some(seen)) }, S_OK);
    assert!(found.is_empty());
    // SAFETY: this test's own locals, valid for the call.
    unsafe { close_container(this, container) };
    // SAFETY: this test's own locals, valid for the call.
    unsafe { close_provider(this, provider) };
}

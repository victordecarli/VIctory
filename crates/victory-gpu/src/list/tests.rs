use super::*;
use crate::object::slot;

#[test]
fn every_inline_command_falls_back_to_the_table() {
    let null = std::ptr::null();
    let mut allocator = null;
    // SAFETY: this test's own locals, valid for the call.
    assert_eq!(unsafe { create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator) }, S_OK);
    let allocator = crate::test_support::valid(allocator);
    let mut list = null;
    // SAFETY: this test's own locals, valid for the call.
    let hr = unsafe { create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list) };
    assert_eq!(hr, S_OK);
    let list = crate::test_support::valid(list);
    // What the title's inline draw checks: put < draw limit.
    // SAFETY: the list is 0x560 bytes.
    let (put, draw_limit, api_limit) = unsafe {
        let at = |o: usize| list.cast::<u8>().add(o).cast::<u64>().read();
        (at(0x18), at(0x28), at(0x30))
    };
    assert!(put >= draw_limit && put >= api_limit);
    // The console's ID3D12GraphicsCommandList6, as Grand Theft Auto VI asks.
    let mut same = null;
    // SAFETY: this test's own locals, valid for the call.
    assert_eq!(unsafe { list_query(list, &LIST_IIDS[7], &mut same) }, S_OK);
    assert_eq!(same, list);
    assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x10)(list), 1);
    assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x40)(list), 0);
    assert_eq!(slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list), S_OK);
    // List-owned upload memory: aligned, writable, gone at reset.
    // SAFETY: this test's own locals, valid for the call.
    let p = unsafe { allocate_upload(list, 40, 0x100) };
    assert!(!p.is_null() && (p as usize).is_multiple_of(0x100));
    let p = crate::test_support::valid_mut(p);
    // SAFETY: 40 bytes were handed out.
    unsafe { p.write_bytes(0xab, 40) };
    // SAFETY: this test's own locals, valid for the call.
    assert!(unsafe { allocate_upload(list, 8, 0x30) }.is_null(), "not a power of two");
    // SAFETY: this test's own locals, valid for the call.
    assert_eq!(unsafe { reset(list, allocator, null) }, S_OK);
    // SAFETY: the list is ours.
    assert!(unsafe { &(*list.cast::<List>()).uploads }.lock().unwrap().is_empty());
    let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
    assert_eq!(release(list), 0);
    assert_eq!(release(allocator), 0);
}

#[test]
fn an_indirect_execution_without_a_signature_is_counted_as_dropped() {
    let null = std::ptr::null();
    // SAFETY: null list, signature and buffers, which the contract allows.
    unsafe { execute_indirect(null, null, 1, null, 0, null, 0) };
    assert_eq!(take_indirect(), 1);
}

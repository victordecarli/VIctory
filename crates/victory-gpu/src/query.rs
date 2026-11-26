//! Query heaps. Nothing runs on a GPU yet, so no query ever produces a
//! result; the heap only remembers its description.

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUERY_HEAP, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const HEAP_IIDS: [Iid; 4] = [
    guid(0x0d9658ae, 0xed45, 0x469e, [0xa6, 0x1d, 0x97, 0x0e, 0xc5, 0x83, 0xca, 0xb4]),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Desc {
    kind: u32,
    count: u32,
    nodes: u32,
}

#[repr(C)]
struct Heap {
    object: Object,
    desc: Desc,
}

/// Device method.
///
/// # Safety
///
/// `desc` is null or points to a query heap description valid for reads for
/// the duration of the call, as `CreateQueryHeap` requires. `iid` is null
/// or points to an interface id valid for reads, and `out` is null or valid
/// for a write of an object pointer, for the duration of the call.
pub(crate) unsafe extern "system" fn create(
    _: This,
    desc: *const Desc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: `desc` is null or readable, per this function's contract.
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "query heap");
    let heap = Box::new(Heap { object: Object::new(&FUNCTIONS), desc });
    // SAFETY: the pointers passed on are under this function's own contract.
    unsafe { object::hand_out(QUERY_HEAP, heap, iid, &HEAP_IIDS, out) }
}

/// `QueryInterface`.
///
/// # Safety
///
/// `this` is a live query heap this module created. `iid` is null or points
/// to an interface id valid for reads, and `out` is null or valid for a
/// write of an object pointer, for the duration of the call.
unsafe extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    // SAFETY: the pointers passed on are under this function's own contract.
    unsafe { object::answer(QUERY_HEAP, iid, object::is(iid, &HEAP_IIDS).then_some(this), out) }
}
/// `Release`.
///
/// # Safety
///
/// `this` is a live query heap this module created, holding the reference
/// the caller gives up; it must not be used after the last one is released.
unsafe extern "system" fn release(this: This) -> u32 {
    // SAFETY: a live object this module made with `hand_out` from a box of
    // this type (this function's contract).
    unsafe { object::release_boxed::<Heap>(this) }
}

/// IUnknown 3, object 4, device child 1.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<QUERY_HEAP, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table(f)
};

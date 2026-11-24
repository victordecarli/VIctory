//! Command signatures: the record layout of indirect execution. Kept for
//! translating it later.

use crate::object::{
    self, COMMAND_SIGNATURE, E_INVALIDARG, HRESULT, Iid, Object, Table, This, guid,
};
use crate::resource::PAGEABLE_IIDS;

const SIGNATURE_IIDS: [Iid; 4] = [
    guid(0xc36a797c, 0xec80, 0x4f0a, [0x89, 0x85, 0xa7, 0xb2, 0x47, 0x50, 0x82, 0xd1]),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
pub(crate) struct Desc {
    stride: u32,
    count: u32,
    /// Each a type and up to three 32-bit fields.
    arguments: *const [u32; 4],
    nodes: u32,
}

#[repr(C)]
struct Signature {
    object: Object,
    stride: u32,
    arguments: Vec<[u32; 4]>,
}

/// Device method.
///
/// # Safety
///
/// `desc` is null or points to a command signature description valid for
/// reads for the duration of the call, as `CreateCommandSignature`
/// requires; when its `count` is non-zero, its `arguments` is null or
/// points to `count` argument descriptions valid for reads for the same
/// duration. `iid` and `out` are passed on to `object::hand_out` unchanged.
pub(crate) unsafe extern "system" fn create(
    _: This,
    desc: *const Desc,
    _root: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: `desc` is null or readable, per this function's contract.
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if desc.arguments.is_null() && desc.count > 0 {
        return E_INVALIDARG;
    }
    let arguments = if desc.count == 0 {
        Vec::new()
    } else {
        // SAFETY: `arguments` is non-null here and points to `count`
        // readable descriptions, per this function's contract.
        unsafe { std::slice::from_raw_parts(desc.arguments, desc.count as usize) }.to_vec()
    };
    tracing::debug!(stride = desc.stride, ?arguments, "command signature");
    let signature =
        Box::new(Signature { object: Object::new(&FUNCTIONS), stride: desc.stride, arguments });
    object::hand_out(COMMAND_SIGNATURE, signature, iid, &SIGNATURE_IIDS, out)
}

/// A signature's record stride and arguments (type, then up to three
/// fields each).
///
/// # Safety
///
/// `this` is null or a signature this module created with `create` and the
/// title has not released, and it stays alive for `'a`.
pub(crate) unsafe fn layout<'a>(this: This) -> Option<(u32, &'a [[u32; 4]])> {
    if this.is_null() {
        return None;
    }
    // SAFETY: `this` is one of our live signatures, per this function's
    // contract.
    let s = unsafe { &*this.cast::<Signature>() };
    Some((s.stride, &s.arguments))
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &SIGNATURE_IIDS).then_some(this);
    object::answer(COMMAND_SIGNATURE, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Signature>(this)
}

/// IUnknown 3, object 4, device child 1.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<COMMAND_SIGNATURE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table(f)
};

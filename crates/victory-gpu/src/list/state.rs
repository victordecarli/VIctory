use super::{This, host_list, with_state};
use crate::draw::{IndexView, VertexView};
use crate::warn::warn_once;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT;

/// `count` items at `items`, or none.
///
/// # Safety
///
/// `items` is null or valid for reads of `count` items, which stay valid
/// for `'a`.
unsafe fn items<'a, T>(items: *const T, count: u32) -> &'a [T] {
    if items.is_null() || count == 0 {
        return &[];
    }
    // SAFETY: non-null, and `count` items, per this function's contract.
    unsafe { std::slice::from_raw_parts(items, count as usize) }
}

/// # Safety
///
/// `this` is null or a live list this module created.
pub(super) unsafe extern "system" fn topology(this: This, topology: u32) {
    // SAFETY: as this function's contract gives them.
    unsafe { with_state(this, |list, s| s.set_topology(list, topology)) };
}
/// # Safety
///
/// `this` is null or a live list this module created, and `viewports` is
/// null or valid for reads of `count` viewports.
pub(super) unsafe extern "system" fn viewports(
    this: This,
    count: u32,
    viewports: *const D3D12_VIEWPORT,
) {
    // SAFETY: as this function's contract gives them.
    if let Some(list) = unsafe { host_list(this) } {
        // SAFETY: as this function's contract gives them.
        crate::draw::viewports(list, unsafe { items(viewports, count) });
    }
}
/// # Safety
///
/// `this` is null or a live list this module created, and `rects` is null
/// or valid for reads of `count` rectangles.
pub(super) unsafe extern "system" fn scissors(this: This, count: u32, rects: *const RECT) {
    // SAFETY: as this function's contract gives them.
    if let Some(list) = unsafe { host_list(this) } {
        // SAFETY: as this function's contract gives them.
        crate::draw::scissors(list, unsafe { items(rects, count) });
    }
}
/// # Safety
///
/// `this` is null or a live list this module created, and `factor` is null
/// or valid for reads of four `f32`.
pub(super) unsafe extern "system" fn blend_factor(this: This, factor: *const [f32; 4]) {
    // SAFETY: as this function's contract gives them.
    if let Some(list) = unsafe { host_list(this) } {
        // SAFETY: `factor` is null or readable, per this function's contract.
        crate::draw::blend_factor(list, unsafe { factor.as_ref() });
    }
}
/// # Safety
///
/// `this` is null or a live list this module created.
pub(super) unsafe extern "system" fn stencil_ref(this: This, reference: u32) {
    // SAFETY: as this function's contract gives them.
    if let Some(list) = unsafe { host_list(this) } {
        crate::draw::stencil_ref(list, reference);
    }
}
/// # Safety
///
/// `this` is null or a live list this module created, and `heaps` is null
/// or valid for reads of `count` descriptor heaps this crate created.
pub(super) unsafe extern "system" fn heaps(this: This, count: u32, heaps: *const This) {
    // SAFETY: as this function's contract gives them.
    unsafe { with_state(this, |_, s| s.set_heaps(items(heaps, count))) };
}
/// # Safety
///
/// `this` is null or a live list this module created.
pub(super) unsafe extern "system" fn compute_table(this: This, slot: u32, handle: u64) {
    // SAFETY: as this function's contract gives them.
    unsafe { with_state(this, |_, s| s.set_table(true, slot, handle)) };
}
/// # Safety
///
/// `this` is null or a live list this module created.
pub(super) unsafe extern "system" fn graphics_table(this: This, slot: u32, handle: u64) {
    // SAFETY: as this function's contract gives them.
    unsafe { with_state(this, |_, s| s.set_table(false, slot, handle)) };
}
/// # Safety
///
/// `this` is null or a live list this module created.
pub(super) unsafe extern "system" fn compute_constant(
    this: This,
    slot: u32,
    value: u32,
    offset: u32,
) {
    // SAFETY: as this function's contract gives them.
    unsafe { with_state(this, |_, s| s.set_constants(true, slot, &[value], offset)) };
}
/// # Safety
///
/// `this` is null or a live list this module created.
pub(super) unsafe extern "system" fn graphics_constant(
    this: This,
    slot: u32,
    value: u32,
    offset: u32,
) {
    // SAFETY: as this function's contract gives them.
    unsafe { with_state(this, |_, s| s.set_constants(false, slot, &[value], offset)) };
}
/// # Safety
///
/// `this` is null or a live list this module created, and `values` is null
/// or valid for reads of `count` `u32`.
pub(super) unsafe extern "system" fn compute_constants(
    this: This,
    slot: u32,
    count: u32,
    values: *const u32,
    offset: u32,
) {
    // SAFETY: as this function's contract gives them.
    unsafe { with_state(this, |_, s| s.set_constants(true, slot, items(values, count), offset)) };
}
/// # Safety
///
/// `this` is null or a live list this module created, and `values` is null
/// or valid for reads of `count` `u32`.
pub(super) unsafe extern "system" fn graphics_constants(
    this: This,
    slot: u32,
    count: u32,
    values: *const u32,
    offset: u32,
) {
    // SAFETY: as this function's contract gives them.
    unsafe { with_state(this, |_, s| s.set_constants(false, slot, items(values, count), offset)) };
}
/// Root constant buffer, shader resource and unordered access views.
///
/// # Safety
///
/// `this` is null or a live list this module created.
pub(super) unsafe extern "system" fn compute_view(this: This, slot: u32, address: u64) {
    // SAFETY: as this function's contract gives them.
    unsafe { with_state(this, |_, s| s.set_view(true, slot, address)) };
}
/// # Safety
///
/// `this` is null or a live list this module created.
pub(super) unsafe extern "system" fn graphics_view(this: This, slot: u32, address: u64) {
    // SAFETY: as this function's contract gives them.
    unsafe { with_state(this, |_, s| s.set_view(false, slot, address)) };
}
/// # Safety
///
/// `this` is null or a live list this module created, and `view` is null or
/// valid for reads of an `IndexView`.
pub(super) unsafe extern "system" fn index_buffer(this: This, view: *const IndexView) {
    // SAFETY: `this` as this function's contract gives it, and `view` is null
    // or readable.
    unsafe { with_state(this, |_, s| s.set_index_buffer(view.as_ref())) };
}
/// # Safety
///
/// `this` is null or a live list this module created, and `views` is null
/// or valid for reads of `count` `VertexView` entries.
pub(super) unsafe extern "system" fn vertex_buffers(
    this: This,
    start: u32,
    count: u32,
    views: *const VertexView,
) {
    // SAFETY: as this function's contract gives them.
    unsafe {
        with_state(this, |_, s| {
            if views.is_null() {
                s.clear_vertex_buffers(start, count);
            } else {
                s.set_vertex_buffers(start, items(views, count));
            }
        })
    };
}
/// `(count, handles, one range, depth)`: with one range, the handles
/// follow the first at the render target view size (128 bytes).
///
/// # Safety
///
/// `this` is null or a live list this module created, `handles` is null or
/// valid for reads of `count` descriptor handles (one if `one_range` is
/// set), and `depth` is null or valid for a read of one.
pub(super) unsafe extern "system" fn targets(
    this: This,
    count: u32,
    handles: *const u64,
    one_range: i32,
    depth: *const u64,
) {
    let handles: Vec<u64> = if one_range != 0 && !handles.is_null() {
        // SAFETY: non-null, and a first handle, per this function's contract.
        let first = unsafe { *handles };
        (0..u64::from(count)).map(|i| first + 128 * i).collect()
    } else {
        // SAFETY: as this function's contract gives them.
        unsafe { items(handles, count) }.to_vec()
    };
    // SAFETY: `depth` is null or readable, per this function's contract.
    let depth = unsafe { depth.as_ref() }.copied().unwrap_or(0);
    // SAFETY: as this function's contract gives them.
    unsafe { with_state(this, |list, s| s.set_targets(list, &handles, depth)) };
}
/// # Safety
///
/// `this` is null or a live list this module created.
pub(super) unsafe extern "system" fn draw(
    this: This,
    vertices: u32,
    instances: u32,
    first: u32,
    first_instance: u32,
) {
    super::count_draw();
    // SAFETY: as this function's contract gives them.
    unsafe { with_state(this, |list, s| s.draw(list, vertices, instances, first, first_instance)) };
}
/// # Safety
///
/// `this` is null or a live list this module created.
pub(super) unsafe extern "system" fn dispatch(this: This, x: u32, y: u32, z: u32) {
    super::count_dispatch();
    // SAFETY: as this function's contract gives them.
    unsafe { with_state(this, |list, s| s.dispatch(list, x, y, z)) };
}
/// `DispatchMesh(x, y, z)` (the PC method of List6); the title's mesh
/// programs take `x` groups only.
///
/// # Safety
///
/// `this` is null or a live list this module created.
pub(super) unsafe extern "system" fn dispatch_mesh(this: This, x: u32, y: u32, z: u32) {
    super::count_draw();
    if y * z != 1 {
        warn_once!("a mesh dispatch with y or z groups", x, y, z);
    }
    // SAFETY: as this function's contract gives them.
    unsafe { with_state(this, |list, s| s.dispatch_mesh(list, x, 0)) };
}
/// `DispatchMeshX(x, flags, indirect arguments, payload)`.
/// shortcut: dispatches whose group count is in GPU memory (indirect)
/// are dropped; make them ExecuteIndirect when the title uses them.
///
/// # Safety
///
/// `this` is null or a live list this module created.
pub(super) unsafe extern "system" fn dispatch_mesh_x(
    this: This,
    x: u32,
    _flags: u64,
    indirect: u64,
    _payload: u64,
) {
    super::count_draw();
    // SAFETY: as this function's contract gives them.
    unsafe { with_state(this, |list, s| s.dispatch_mesh(list, x, indirect)) };
}
/// # Safety
///
/// `this` is null or a live list this module created.
pub(super) unsafe extern "system" fn draw_indexed(
    this: This,
    indices: u32,
    instances: u32,
    first: u32,
    base: i32,
    first_instance: u32,
) {
    super::count_draw();
    // SAFETY: as this function's contract gives them.
    unsafe {
        with_state(this, |list, s| {
            s.draw_indexed(list, indices, instances, first, base, first_instance)
        })
    };
}

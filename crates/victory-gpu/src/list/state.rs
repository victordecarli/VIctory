use super::{This, host_list, with_state};
use crate::draw::{IndexView, VertexView};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT;

/// `count` items at `items`, or none.
fn items<'a, T>(items: *const T, count: u32) -> &'a [T] {
    if items.is_null() || count == 0 {
        return &[];
    }
    // SAFETY: the title passes `count` items.
    unsafe { std::slice::from_raw_parts(items, count as usize) }
}

pub(super) extern "system" fn topology(this: This, topology: u32) {
    with_state(this, |list, s| s.set_topology(list, topology));
}
pub(super) extern "system" fn viewports(this: This, count: u32, viewports: *const D3D12_VIEWPORT) {
    if let Some(list) = host_list(this) {
        crate::draw::viewports(list, items(viewports, count));
    }
}
pub(super) extern "system" fn scissors(this: This, count: u32, rects: *const RECT) {
    if let Some(list) = host_list(this) {
        crate::draw::scissors(list, items(rects, count));
    }
}
pub(super) extern "system" fn blend_factor(this: This, factor: *const [f32; 4]) {
    if let Some(list) = host_list(this) {
        // SAFETY: the title passes four floats or null.
        crate::draw::blend_factor(list, unsafe { factor.as_ref() });
    }
}
pub(super) extern "system" fn stencil_ref(this: This, reference: u32) {
    if let Some(list) = host_list(this) {
        crate::draw::stencil_ref(list, reference);
    }
}
pub(super) extern "system" fn heaps(this: This, count: u32, heaps: *const This) {
    with_state(this, |_, s| s.set_heaps(items(heaps, count)));
}
pub(super) extern "system" fn compute_table(this: This, slot: u32, handle: u64) {
    with_state(this, |_, s| s.set_table(true, slot, handle));
}
pub(super) extern "system" fn graphics_table(this: This, slot: u32, handle: u64) {
    with_state(this, |_, s| s.set_table(false, slot, handle));
}
pub(super) extern "system" fn compute_constant(this: This, slot: u32, value: u32, offset: u32) {
    with_state(this, |_, s| s.set_constants(true, slot, &[value], offset));
}
pub(super) extern "system" fn graphics_constant(this: This, slot: u32, value: u32, offset: u32) {
    with_state(this, |_, s| s.set_constants(false, slot, &[value], offset));
}
pub(super) extern "system" fn compute_constants(
    this: This,
    slot: u32,
    count: u32,
    values: *const u32,
    offset: u32,
) {
    with_state(this, |_, s| s.set_constants(true, slot, items(values, count), offset));
}
pub(super) extern "system" fn graphics_constants(
    this: This,
    slot: u32,
    count: u32,
    values: *const u32,
    offset: u32,
) {
    with_state(this, |_, s| s.set_constants(false, slot, items(values, count), offset));
}
/// Root constant buffer, shader resource and unordered access views.
pub(super) extern "system" fn compute_view(this: This, slot: u32, address: u64) {
    with_state(this, |_, s| s.set_view(true, slot, address));
}
pub(super) extern "system" fn graphics_view(this: This, slot: u32, address: u64) {
    with_state(this, |_, s| s.set_view(false, slot, address));
}
pub(super) extern "system" fn index_buffer(this: This, view: *const IndexView) {
    // SAFETY: the title passes a view or null.
    with_state(this, |_, s| s.set_index_buffer(unsafe { view.as_ref() }));
}
pub(super) extern "system" fn vertex_buffers(
    this: This,
    start: u32,
    count: u32,
    views: *const VertexView,
) {
    with_state(this, |_, s| {
        if views.is_null() {
            s.clear_vertex_buffers(start, count);
        } else {
            s.set_vertex_buffers(start, items(views, count));
        }
    });
}
/// `(count, handles, one range, depth)`: with one range, the handles
/// follow the first at the render target view size (128 bytes).
pub(super) extern "system" fn targets(
    this: This,
    count: u32,
    handles: *const u64,
    one_range: i32,
    depth: *const u64,
) {
    let handles: Vec<u64> = if one_range != 0 && !handles.is_null() {
        // SAFETY: the title passes the first handle.
        let first = unsafe { *handles };
        (0..u64::from(count)).map(|i| first + 128 * i).collect()
    } else {
        items(handles, count).to_vec()
    };
    // SAFETY: the title passes a handle or null.
    let depth = unsafe { depth.as_ref() }.copied().unwrap_or(0);
    with_state(this, |list, s| s.set_targets(list, &handles, depth));
}
pub(super) extern "system" fn draw(
    this: This,
    vertices: u32,
    instances: u32,
    first: u32,
    first_instance: u32,
) {
    super::count_draw();
    with_state(this, |list, s| s.draw(list, vertices, instances, first, first_instance));
}
pub(super) extern "system" fn dispatch(this: This, x: u32, y: u32, z: u32) {
    super::count_dispatch();
    with_state(this, |list, s| s.dispatch(list, x, y, z));
}
/// `DispatchMesh(x, y, z)` (the PC method of List6); the title's mesh
/// programs take `x` groups only.
pub(super) extern "system" fn dispatch_mesh(this: This, x: u32, y: u32, z: u32) {
    super::count_draw();
    if y * z != 1 {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(x, y, z, "a mesh dispatch with y or z groups"));
    }
    with_state(this, |list, s| s.dispatch_mesh(list, x));
}
/// `DispatchMeshX(x, flags, indirect arguments, payload)`.
/// shortcut: dispatches whose group count is in GPU memory (indirect)
/// are dropped; make them ExecuteIndirect when the title uses them.
pub(super) extern "system" fn dispatch_mesh_x(
    this: This,
    x: u32,
    _flags: u64,
    indirect: u64,
    _payload: u64,
) {
    super::count_draw();
    if indirect != 0 {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!("indirect mesh dispatches dropped"));
        return;
    }
    with_state(this, |list, s| s.dispatch_mesh(list, x));
}
pub(super) extern "system" fn draw_indexed(
    this: This,
    indices: u32,
    instances: u32,
    first: u32,
    base: i32,
    first_instance: u32,
) {
    super::count_draw();
    with_state(this, |list, s| {
        s.draw_indexed(list, indices, instances, first, base, first_instance)
    });
}

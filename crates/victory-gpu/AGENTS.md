# victory-gpu

What is local to this crate. The root `AGENTS.md` still applies in full.

## How commands arrive

D3D12.X inlines packet writing into the title and calls the list's method
only when its put pointer reaches a limit. Our lists keep every limit at
zero, so each command is a call into a table method here. Title GPU addresses are CPU addresses (unified memory).

Every table method is title-reachable: on any failure it logs and returns
the API's failure code, or drops the draw. Shader translation failing at
pipeline creation is one such failure.

A dropped draw, mesh dispatch or dispatch is counted twice: on `DROPPED`
in `crates/victory-gpu/src/draw.rs` (the frame report's total) and on its
cause, through `pipeline::count_drop`, or `pipeline::drops::general` with
a `Cause` when no pipeline is to blame. A drop missing either is missing
from that report in the run log.

## Objects and tables

Each object starts with a function-table pointer, bookkeeping bits and a
reference count (`object`); objects are static and never destroyed.
Tables come from `missing_table`, longer than any interface we know, so a
method a newer title adds logs and fails instead of jumping past the
table. Fill a slot by writing the `extern "system"` method and putting it
in that slot.

A method that is not written yet, or a case it skips, logs once (a
`std::sync::Once` or the file's `once!`) with enough context that a run
lists what to do next.

## Title memory on the host

The host GPU cannot read title memory. `shadow` keeps a host copy of each
buffer a draw uses and refreshes the ranges a submission reads from the
pages the title wrote (`watch`, fed by write watching that `victory-hle`
`xmem` sets up). Shader writes go back to title memory after the
submission. A new way for a draw to read a buffer needs its range added to
what `shadow` refreshes, or the draw reads the first copy forever.

## Linux

The crate compiles on Linux, but `draw`, `host`, `shadow`, `watch`,
`shader_cache` and the rest of the host side are `#[cfg(windows)]`. A
portable module (`list`, `resource`, `pipeline` ...) that calls into them
must be gated too, or Linux stops building (as `resource::view_format`
did). Helpers that only Windows code uses belong under `#[cfg(windows)]`:
they are why Linux clippy still excludes this crate.

## Pipelines

`crates/victory-gpu/src/pipeline/host.rs` holds a stage together with its translation from
construction (`Stage`), so nothing later has to unwrap a translation that
might be missing. Keep new pipeline state in that shape: build it fallibly
once, then carry the typed value. A host pipeline's root signature is not
built here: `HostRootLayout` (`crates/victory-shader/src/emit/root_layout.rs`)
decides it and `crates/victory-shader/src/emit/host/root.rs` creates it, the
same code the GPU tests use. A `Stage` keeps its `HostRootStage`, which says
where its user data and root views went.

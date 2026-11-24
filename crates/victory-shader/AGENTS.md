# victory-shader

What is local to this crate. The root `AGENTS.md` still applies in full.

## Stages

`translate` in `emit` runs the whole pipeline:

```text
dxbc / dxil   container, XDAT table, zlib-compressed GFX10 code
ir::build     GFX10 decode (gfx10, native) to SSA
structure     EXEC and lane masks become per-lane booleans; unproven EXEC stays a per-lane guard
resource      where each descriptor and address comes from
io            inputs and outputs raised from the native layout
emit          DXIL in a PC container (SM 6.6) plus the HostShader binding contract
```

`emit`'s module doc is the binding contract `victory-gpu` relies on (user
data, descriptor heaps, static samplers, root CBVs/SRVs, vertex fetch).
Change it in the same commit as the code it describes.

## Purity

The crate is pure except `emit::host` (Windows, test or feature `host`),
`emit::validator` (Windows) and the `VICTORY_WARP` read in `emit::host`.
Code that needs D3D12, DXC or the validator goes in those two modules;
everything else builds and tests on Linux.

## Failure is a value

Translation runs inside `victory-gpu` pipeline creation, on the title's
call, so every failure is a `ShaderError`:

- `Unsupported { offset, what }`: valid code we do not translate yet.
- `Isa`, `Malformed`, `NotDxbc`: bad input.
- `Internal`: we broke our own invariant. The bitcode builder records such
  breaks on the `Module` and `write` returns them; follow that shape.

Inside `lower`, a value the emitter cannot express is `Missing` with a
reason. It fails the program only when something with an effect (an
export, a branch) needs it, so an unused unsupported value costs nothing.

## lower.rs

`crates/victory-shader/src/emit/lower.rs` is many times the file limit and
holds most of the crate's `unreachable!`. Put a new lowering in its own
module under `emit/` and call it from `lower`, as `wave` is. An `unreachable!` you touch there becomes a `let ... else` returning
the lowering's error.

## Tests

- Portable unit tests sit beside the code under `#[cfg(test)]`.
- `diff_tests`, `valu_tests` and `waterfall_tests` run translated shaders
  on the host GPU: Windows only, gated `#[cfg(test)]` plus
  `#[cfg(windows)]`.
- `diff_tests` compile each HLSL case with the console and the PC
  compiler and compare the two. They skip unless
  `VICTORY_CONSOLE_DXC` names the console compiler; `VICTORY_PC_DXC`
  overrides the PC one (default: the `dxc.exe` beside the validator),
  `VICTORY_DIFF_ONLY` filters cases, `VICTORY_EMIT_DUMP` names a directory
  for emitted modules, `VICTORY_WARP=1` runs on WARP.

## HostShader is persisted

`victory-gpu`'s `shader_cache` stores `HostShader` as `VSH1` bytes. A new
field needs its `encode` and `decode` updated in the same change, or
cached shaders lose it silently.

---
name: ratchet-ffi
description: Removes one module's not_unsafe_ptr_arg_deref allow by making its functions that dereference title pointers unsafe. One module per run.
disable-model-invocation: true
argument-hint: "[module path]"
---

# Ratchet FFI

Pays down one module of the `not_unsafe_ptr_arg_deref` debt in the root
`AGENTS.md` (hard rule 3, Lint ratchet): one module per run, no change in
behaviour, verified on Windows.

## 1. Preconditions

- The host is Windows. Elsewhere, stop and say so: the ratchet requires
  Windows verification, and `wincheck` does not run tests.
- The tree is clean, on a branch off `main`.

## 2. Pick the module

The argument, if given. Otherwise the smallest file among
`git grep -l not_unsafe_ptr_arg_deref -- crates/`.

Skip `crates/victory-gpu/src/object.rs` and
`crates/victory-hle/src/modules/xgameruntime.rs` unless asked by name:
they hold the shared pointer helpers (`hand_out`, `answer`,
`release_boxed`; `put`, `text`), so converting them reaches every module
that calls them.

## 3. Find every target

The lint is a lower bound: it fires only on exported `pub` functions.
`pub(crate)` and private methods reached through function tables pass it
while still dereferencing title pointers (`create` in
`crates/victory-gpu/src/query.rs` does, and clippy stays clean without
the allow). Hard rule 3 is the bar.

List every function in the module with a raw-pointer parameter and mark
each:

- **convert**: its own body dereferences the pointer (`*p`, `as_ref`,
  `as_mut`, `read`, `write`, `slice::from_raw_parts`, `CStr::from_ptr`)
  or passes it to an `unsafe` callee.
- **keep**: it only forwards the pointer to a helper in another module,
  null-checks it, or compares it.

The step is done when the list covers every pointer-taking function in
the file. Show it before editing.

## 4. Convert

For each **convert**:

- Make it `unsafe extern "system" fn` (or `unsafe fn`), keeping its
  visibility.
- Add a `# Safety` section with the caller's contract, from the API it
  implements: which pointers must be non-null, valid for reads or writes,
  of what length, and for how long.
- Keep each `unsafe {}` block (edition 2024 with `unsafe_op_in_unsafe_fn`
  denied still requires them) and make its `// SAFETY:` comment point at
  the `# Safety` contract.
- Function tables (`as *const ()`) and `hle_export!` take an unsafe fn
  unchanged. A Rust caller wraps the call in `unsafe {}` with a
  `// SAFETY:` naming where its pointer comes from.

Then delete the module's `#![allow(clippy::not_unsafe_ptr_arg_deref, ...)]`.
The diff holds signatures, docs and safety comments only; any logic edit
belongs in another change.

## 5. Verify and record

- Run the `verify` skill; both Windows clippy runs and the tests must be
  clean.
- Update every count of these allows in the root `AGENTS.md` in the same
  commit (Pitfalls and Lint ratchet: the total and the per-crate split).
  `git grep -l not_unsafe_ptr_arg_deref -- crates/<crate> | wc -l` gives
  each crate's new number.
- One commit: `refactor(<crate>): make <module> pointer functions unsafe`.

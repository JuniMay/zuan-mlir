# Kernel Building System

This document describes Dyno's CMake-side kernel building system: how MLIR
kernel sources are lowered, compiled into target objects, linked into
benchmarks or runtime regressions, and executed in native and cross builds.

The implementation lives primarily in:

- `cmake/DynoHostTools.cmake`
- `cmake/DynoKernelBuild.cmake`
- `cmake/DynoKernelLowering.cmake`
- `cmake/DynoBenchmarkKernels.cmake`
- `cmake/DynoRegressionKernels.cmake`
- `cmake/DynoGoogleBenchmark.cmake`
- `cmake/DynoTriton.cmake`
- `cmake/DynoRewriteText.cmake`

## Scope

Dyno currently has two kernel-consuming layers:

1. `benchmarks/`
2. `regression/`

Both layers share the same lowering and object-generation helpers. The
benchmark layer adds benchmark-specific packaging such as Google Benchmark. The
runtime regression layer adds driver executables and `CTest` registration.

## High-Level Flow

For a typical Dyno or Linalg kernel, the build flow is:

1. Optionally rewrite the MLIR entry symbol name.
2. Lower MLIR to an LLVM-dialect `.mlir` file.
3. Translate the LLVM-dialect `.mlir` file to textual LLVM IR `.ll`.
4. Compile the LLVM IR to a target object `.o`.
5. Package the object as a static library or link it into an executable.
6. For runtime regressions, register a `CTest` entry that executes the linked
   binary, using QEMU when cross-compiling.

The generated intermediate files are usually named with a common prefix:

- `<prefix>.entry.mlir`: rewritten entry-point input, if entry renaming is used
- `<prefix>.mlir`: lowered LLVM-dialect MLIR
- `<prefix>.mlir.ir/`: optional pass-by-pass IR dump tree rooted in the
  corresponding build directory
- `<prefix>.ll`: translated LLVM IR
- `<prefix>.o`: compiled object

## Host Tools vs Target Artifacts

The build system distinguishes:

- host-side lowering tools
- target-side executables and objects

Host-side tools are resolved by `dyno_configure_kernel_build()` from
`cmake/DynoKernelBuild.cmake`, which delegates to
`dyno_configure_host_tools()` in `cmake/DynoHostTools.cmake`. These include:

- `mlir-opt`
- `mlir-translate`
- `clang`
- `dyno-opt`
- `dyno-translate`

Resolution is explicit-path-first:

- `DYNO_HOST_LLVM_TOOLS_DIR`
- `DYNO_HOST_TOOLS_DIR`
- in-tree `dyno-opt` / `dyno-translate` targets when they are available
- derived LLVM bin directories such as `${LLVM_BINARY_DIR}/bin`
- the active `PATH` as a final fallback

`clang` is not searched independently. It comes directly from
`CMAKE_C_COMPILER`, so the configured C compiler is also the compiler used for
kernel object generation.

In the normal scripted RISC-V flow, `scripts/build-env.sh` and
`scripts/build-riscv-dyno.sh` already pass explicit `DYNO_HOST_*` paths, so the
helper layer typically does not need to fall back to search at all.

Native builds are more permissive: if the explicit host-tool directories are
not provided, CMake may still use in-tree tool targets, derived LLVM bin
directories, or `PATH` lookups.

Cross builds are stricter than that native path. They now require
`DYNO_HOST_LLVM_TOOLS_DIR` and `DYNO_HOST_TOOLS_DIR` to be set explicitly
instead of attempting implicit discovery.

Target-built executables are wrapped by `dyno_wrap_target_command()`. When
`CMAKE_CROSSCOMPILING` and `CMAKE_CROSSCOMPILING_EMULATOR` are set, the wrapper
uses:

- `CMAKE_CROSSCOMPILING_EMULATOR`
- `DYNO_QEMU_LD_PREFIX`
- `DYNO_QEMU_CPU`

This keeps lowering on the host while still allowing runtime artifacts to be
executed through QEMU.

## Entry-Point Rewriting

Many helpers expect the source kernel entry to be spelled as:

```mlir
func.func @kernel(...)
```

`dyno_rewrite_mlir_kernel()` rewrites the textual token `kernel(` to a new
entry name using `cmake/DynoRewriteText.cmake`.

This is how the benchmark and regression helpers generate stable wrapper names
such as:

- `kernel_scalar`
- `kernel_dyno_16_2`
- `kernel_transform_8_4`
- `kernel_dyno`

Important limitation:

- the rewrite is textual and specifically targets `kernel(`
- kernels that use another symbol name must either be adapted to `@kernel` or
  use a helper path that does not request renaming

## Core Lowering Helpers

`cmake/DynoKernelLowering.cmake` defines the shared lowering functions.

### `dyno_compile_llvm_ir_to_object`

Compiles LLVM IR to an object file using the configured host `clang`.

Key points:

- always adds `-march=rv64gcv_zvfh`
- always includes the target/sysroot flags derived from `CMAKE_C_FLAGS`
- accepts additional optimization or debug flags through `EXTRA_FLAGS`

### `dyno_lower_scalar_mlir_to_llvm_ir`

Lowers a scalar Linalg kernel with `mlir-opt` and `mlir-translate`.

Typical pipeline shape:

- optional bufferization
- Linalg to affine loops
- canonicalization and CSE
- LLVM lowering
- optional `-llvm-request-c-wrappers`

### `dyno_lower_dyno_mlir_to_llvm_ir`

Lowers a Dyno kernel, or a Linalg kernel through Dyno, with `dyno-opt` and
`dyno-translate`.

This is the central helper for Dyno-backed kernels.

It supports:

- `INPUT_IS_DYNO`
  - skip `-convert-linalg-to-dyno`
- `VF`, `UF`
  - stripmining configuration
- `REDUCTION_MODE`
  - passed into `-dyno-stripmining=... reduction-mode=<...>`
- `FP_POLICY`
  - passed into `-dyno-stripmining=... fp-policy=<...>`
- `FUSE_ELEMENTWISE`
- `STRENGTH_REDUCTION`
- `PRE_VP_CANONICALIZE`
- `POST_VP_CANONICALIZE`
- `FOLD_MEMREF_ALIAS_OPS`
- `REQUEST_C_WRAPPERS`
- `ENABLE_MATH_ESTIMATION`

The generated pass structure is conceptually:

1. optional Linalg-to-Dyno conversion
2. `-lower-dyno=target-rank=2`
3. `-dyno-stripmining=...`
4. optional Dyno cleanups
5. `-convert-dyno-to-vp=...`
6. optional VP-side cleanups
7. `-convert-vp-to-llvm=...`
8. standard LLVM-dialect lowering and cleanup

### `dyno_lower_autovec_mlir_to_llvm_ir`

Builds the auto-vectorized comparison path from Linalg without going through
Dyno.

### `dyno_lower_transform_mlir_to_llvm_ir`

Builds the transform-dialect comparison path.

Its `transform.txt` template is materialized at configure/build time by
replacing the placeholder tokens `VF` and `UF`.

## Benchmark Helper Layer

The benchmark layer is enabled by `ENABLE_BENCHMARKS=ON` and configured from
`benchmarks/CMakeLists.txt`.

It does the following:

1. configures Google Benchmark
2. configures the shared kernel build layer
3. loads `DynoBenchmarkKernels.cmake`
4. adds benchmark subdirectories

The main public helper functions are:

- `add_linalg_scalar_kernel(benchmark_name)`
- `add_linalg_dyno_kernel(benchmark_name vf uf)`
- `add_linalg_autovec_kernel(benchmark_name virtvecsize)`
- `add_linalg_transform_kernel(benchmark_name vf uf)`
- `dyno_add_direct_dyno_kernel(TARGET ... INPUT ... VF ... UF ...)`

These helpers produce static libraries containing the compiled kernel object.
The benchmark's `launcher.cpp` then links against those libraries.

Example benchmark usage:

```cmake
add_linalg_dyno_kernel(matmul 16 2)
target_link_libraries(linalg-matmul-benchmark PRIVATE linalg_matmul_dyno_16_2)
```

## Runtime Regression Helper Layer

The runtime regression layer is enabled by:

```cmake
-DENABLE_RUNTIME_REGRESSIONS=ON
```

and configured from `regression/CMakeLists.txt`.

It:

1. configures the shared kernel build layer
2. loads `DynoRegressionKernels.cmake`
3. adds regression subdirectories
4. creates the convenience target `check-dyno-runtime-regressions`

The public regression helpers are:

```cmake
add_linalg_runtime_regression(
  NAME <name>
  MLIR_FILE <path>
  DRIVER <path>
  DYNO_VF <vf>
  DYNO_UF <uf>
  DYNO_REDUCTION_MODE <parallel|sequential|auto>
  DYNO_FP_POLICY <strict|relaxed>
  EPSILON <eps>)

add_dyno_runtime_regression(
  NAME <name>
  MLIR_FILE <path>
  DRIVER <path>
  DYNO_VF <vf>
  DYNO_UF <uf>
  DYNO_REDUCTION_MODE <parallel|sequential|auto>
  DYNO_FP_POLICY <strict|relaxed>
  EPSILON <eps>)
```

Internally, each regression helper builds:

1. a lowered Dyno kernel object
2. a small driver executable
3. a `CTest` entry labeled with `runtime-regression`

The driver is responsible for:

- deterministic initialization
- reference computation
- comparison and diagnostics

Regression drivers also receive compile-time definitions describing the case:

- `DYNO_REGRESSION_NAME`
- `DYNO_REGRESSION_DYNO_VF`
- `DYNO_REGRESSION_DYNO_UF`
- `DYNO_REGRESSION_EPSILON`
- `DYNO_REGRESSION_REDUCTION_MODE`
- `DYNO_REGRESSION_FP_POLICY`

## Public CMake Option Surface

The main top-level switches are:

- `ENABLE_BENCHMARKS`
- `ENABLE_TRITON_BENCHMARKS`
- `ENABLE_RUNTIME_REGRESSIONS`
- `DYNO_BENCHMARK_DUMP_IR`

`DYNO_BENCHMARK_DUMP_IR` enables MLIR's file-tree IR dumping for benchmark and
runtime-regression lowering.

When enabled, each lowered kernel gets a sibling dump directory rooted at:

- `<output>.mlir.ir/`

The helper layer passes:

- `-mlir-print-ir-after-all`
- `-mlir-print-ir-tree-dir=<output>.mlir.ir`

This writes pass dumps into the kernel's build directory instead of streaming
them to the terminal.

## Common Helper Keyword Semantics

Several helper keywords in `dyno_lower_dyno_mlir_to_llvm_ir()` are boolean
switches parsed with `cmake_parse_arguments(...)`.

Important ones are:

- `STRENGTH_REDUCTION`
  - adds `-dyno-strength-reduction`
- `PRE_VP_CANONICALIZE`
  - adds `-canonicalize -cse` before VP lowering
- `POST_VP_CANONICALIZE`
  - adds `-canonicalize -cse` after VP lowering
- `FOLD_MEMREF_ALIAS_OPS`
  - adds `-fold-memref-alias-ops`
- `REQUEST_C_WRAPPERS`
  - adds `-llvm-request-c-wrappers`

These are local helper keywords. They are not standalone cache variables.

## Triton Path Defaults

`cmake/DynoTriton.cmake` computes most of its internal path graph from a small
set of defaults instead of publishing every intermediate directory as a cache
entry.

The important override points are the leaf tools and Python executable, for
example:

- `DYNO_TRITON_PYTHON_EXECUTABLE`
- `DYNO_TRITON_SHARED_OPT`
- `DYNO_TRITON_LLVM_BINARY_DIR`
- `DYNO_TRITON_CPU_OPT`
- `DYNO_TRITON_CPU_LLVM_BINARY_DIR`

The RISC-V build scripts already pass those explicitly when Triton benchmarks
are enabled.

## Naming and Packaging Conventions

The benchmark helpers follow these naming patterns:

- `linalg_<benchmark>_scalar`
- `linalg_<benchmark>_dyno_<vf>_<uf>`
- `linalg_<benchmark>_transform_<vf>_<uf>`
- `linalg_<benchmark>_autovec_<virtvecsize>`

The runtime regression helpers use:

- `<test-name>_dyno` for the kernel static library
- `<test-name>` for the driver executable and `CTest` entry

## Adding a New Linalg Benchmark Kernel

Typical pattern:

1. add `<name>.mlir`
2. add `launcher.cpp`
3. call one or more benchmark helpers from that directory's `CMakeLists.txt`
4. link the generated kernel libraries into the benchmark executable

Minimal example:

```cmake
add_executable(linalg-foo-benchmark launcher.cpp)
target_link_libraries(linalg-foo-benchmark PRIVATE GoogleBenchmark)

add_linalg_scalar_kernel(foo)
add_linalg_dyno_kernel(foo 16 2)

target_link_libraries(
  linalg-foo-benchmark
  PRIVATE
  linalg_foo_scalar
  linalg_foo_dyno_16_2
)
```

## Adding a New Runtime Regression

Typical pattern:

1. add a small fixed-shape driver
2. use a benchmark MLIR file or add a regression-local MLIR file
3. register the test with `add_linalg_runtime_regression(...)` or
   `add_dyno_runtime_regression(...)`

Minimal example:

```cmake
add_linalg_runtime_regression(
  NAME runtime-linalg-foo
  MLIR_FILE ${DYNO_SOURCE_DIR}/benchmarks/linalg/foo/foo.mlir
  DRIVER ${CMAKE_CURRENT_SOURCE_DIR}/driver.cpp
  DYNO_VF 16
  DYNO_UF 2
  DYNO_REDUCTION_MODE sequential
  DYNO_FP_POLICY strict
  EPSILON 0.0001
)
```

## Cross-Build Expectations

For RISC-V builds, the intended setup is:

- lower with host-built tools
- compile target objects with the configured target flags
- execute runtime tests through QEMU

In practice this means:

- `DYNO_HOST_TOOLS_DIR` may point at a native Dyno build
- `DYNO_HOST_LLVM_TOOLS_DIR` may point at a native LLVM build
- `CMAKE_C_COMPILER` should point at the host-side `clang` used to compile
  target LLVM IR into objects
- `CMAKE_CROSSCOMPILING_EMULATOR` should point at `qemu-riscv64`
- `DYNO_QEMU_LD_PREFIX` should point at the RISC-V sysroot
- `DYNO_QEMU_CPU` may be used to fix the expected RVV configuration

## Current Limitations

- Entry-point rewriting assumes the source symbol is `@kernel`.
- The helper layer is built around MLIR textual pipelines rather than a more
  structured pass-pipeline description.
- The benchmark and regression flows share lowering code by design, so changes
  to `DynoKernelLowering.cmake` affect both layers.

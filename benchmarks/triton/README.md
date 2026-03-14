# Triton Benchmarks

This directory contains several Triton programs for benchmarking purposes. The
current cases are:

- `matmul`
- `attention`
- `vector_add`
- `vector_mul`
- `saxpy`

Each program is compiled twice:

- with `triton-cpu` for the reference CPU lowering
- with `triton_shared` and then through the Zuan lowering pipeline

The shared CMake helper in [CMakeLists.txt](/root/zuan-mlir/benchmarks/triton/CMakeLists.txt)
now also builds a scalar reference kernel from the generated linalg MLIR
through:

- bufferization
- `convert-linalg-to-affine-loops`
- loop/CF lowering
- LLVM lowering

That gives three generated kernel entry points per benchmark:

- `kernel_triton_cpu`
- `kernel_zuan`
- `kernel_scalar`
- `kernel_zuan_wrapper`
- `kernel_scalar_wrapper`

**Kernel Signatures**
`kernel_triton_cpu`

- This is the launch-style ABI already exposed by the generated header.
- It takes raw pointers/scalars first, then the implicit Triton launch
  arguments in this order:
  `x, y, z, gridX, gridY, gridZ`.
- It is meant to be called through `launch_kernel(...)`.

`kernel_zuan` and `kernel_scalar`

- These use the same raw MLIR unranked-memref ABI.
- Each memref argument is lowered as a pair:
  `int64_t rank_i, void *descriptor_i`
- After the memref descriptor pairs come the original scalar kernel arguments
  from the Triton-derived MLIR.
- The implicit launch arguments come last, in this order:
  `gridX, gridY, gridZ, x, y, z`

The generated header also emits:

- `kernel_zuan_wrapper`
- `kernel_scalar_wrapper`
- `kernel_ptr_t`
- `launch_kernel(...)`

So the practical rule is:

- `kernel_triton_cpu` uses the launcher ABI directly
- `kernel_zuan_wrapper` / `kernel_scalar_wrapper` already adapt the raw
  pointer launcher ABI into the MLIR memref ABI and forward the implicit launch
  arguments in `gridX, gridY, gridZ, x, y, z` order

**Adding A New Triton Benchmark**
1. Add `<name>.py` that emits the TTIR and wrapper header.
2. Add a subdirectory with a launcher and call `add_triton_benchmark(<name> "...args...")`.
3. Include the generated `<name>.h` in the launcher for `kernel_triton_cpu`,
   `kernel_zuan_wrapper`, `kernel_scalar_wrapper`, `launch_kernel`, and
   `kernel_ptr_t`.
4. Use `runKernel(kernel_scalar_wrapper, ...)` for reference checking instead
   of handwritten scalar math.

These benchmarks are optional. Enable them with `-DENABLE_TRITON_BENCHMARKS=ON`
after bootstrapping the project-local Triton toolchain:

```bash
./scripts/setup-triton.sh build-riscv
```

The setup script keeps the Triton Python environment, build artifacts, and LLVM
cache under the selected build tree, so disabling Triton support is just a
matter of leaving `ENABLE_TRITON_BENCHMARKS` off.

The script now builds the Triton pieces in two separate roots under the build
tree:

- `shared/`: upstream Triton pinned to the commit expected by `triton_shared`
- `cpu/`: `triton-cpu`, used only for `triton-opt` and the CPU pass pipeline

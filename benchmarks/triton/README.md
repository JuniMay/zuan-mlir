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

# Dyno MLIR

🚧 Work in progress

Dyno is an MLIR-based prototype compiler framework designed for
dynamic sized vector architectures.

## Get Started

1. Clone and initialize the repository:

```bash
git clone <repo-url> dyno-mlir
cd dyno-mlir
git submodule update --init --recursive
```

2. Install dependencies:

```bash
sudo apt install build-essential cmake ninja-build clang lld
```

Optional: to build the RISC-V GNU Toolchain, first apply the workaround patch:

```bash
./scripts/workaround-gnutoolchain.sh
```

Then check the README in `third_party/riscv-gnu-toolchain` to install dependencies.

Note that the GNU Toolchain is by default enabled in the script, and it also
enables an LLVM build to support OpenMP header and libraries, so it will take
longer to build.

3. Compile LLVM & MLIR

```bash
./scripts/build-llvm.sh
```

4. Compile Dyno

```bash
./scripts/build-dyno.sh
```

## Local Development

For normal host-side development, the existing LLVM and Dyno build flow is
unchanged:

```bash
./scripts/build-llvm.sh
./scripts/build-dyno.sh
```

## RISC-V Benchmarking

To build the benchmarks for RISC-V platforms, cross-compile the project.

```bash
source ./scripts/build-env.sh
./scripts/build-riscv-llvm.sh
./scripts/setup-triton.sh build-riscv
./scripts/build-riscv-dyno.sh
```

Set `ENABLE_TRITON_BENCHMARKS=OFF` before `./scripts/build-riscv-dyno.sh` if you
want to skip the Triton benchmarks entirely.

The Triton setup is split on purpose:

- `third_party/triton` is pinned to the commit required by `third_party/triton_shared`
- `third_party/triton_shared` provides the shared lowering and Python backend
- `third_party/triton-cpu` is kept on its own head and is only used for
  `triton-opt`

This avoids having to patch any Triton source tree locally just to make the
benchmarks compile.

## Running Tests and Benchmarks

### IR `lit` tests

Run the compiler IR regression suite from the host build:

```bash
cmake --build build --target check-dyno
```

or use `ninja`

```bash
ninja -C build check-dyno
```

### Runtime regressions

Run the runtime regression suite from the RISC-V build:

```bash
ctest --test-dir build-riscv -L runtime-regression --output-on-failure
```

Or use the convenience target:

```bash
cmake --build build-riscv --target check-dyno-runtime-regressions
```

### Benchmarks

Benchmarks are built as executables under `build-riscv/benchmarks/...`. Run
them directly, for example:

```bash
build-riscv/benchmarks/linalg/matmul/linalg-matmul-benchmark
build-riscv/benchmarks/dyno/matmul/dyno-matmul-benchmark
build-riscv/benchmarks/triton/matmul/triton-matmul-benchmark
```

`qemu-riscv64` may be required to run these benchmarks on a non-RISC-V machine.

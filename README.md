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

If you also want the optional Triton benchmarks on the host, bootstrap the
split Triton toolchain into your host build tree and configure with
`ENABLE_TRITON_BENCHMARKS=ON`:

```bash
./scripts/setup-triton.sh build
cmake -S . -B build -DENABLE_BENCHMARKS=ON -DENABLE_TRITON_BENCHMARKS=ON
ninja -C build triton-vector-add-benchmark triton-vector-mul-benchmark triton-saxpy-benchmark triton-matmul-benchmark
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

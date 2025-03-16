# Zuan MLIR

🚧 Work in progress

Zuan (纂, /tswan/) is an MLIR-based prototype compiler framework designed for
dynamic sized vector architectures.

## Get Started

1. Clone and initialize the repository:

```bash
git clone https://github.com/JuniMay/zuan-mlir
cd zuan-mlir
git submodule update --init
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

4. Compile Zuan

```bash
./scripts/build-zuan.sh
```

5. To build the benchmarks for RISC-V platforms, cross-compile the project.

```bash
source ./scripts/build-env.sh
./scripts/build-riscv-llvm.sh
./scripts/build-riscv-zuan.sh
```

## How it works

Zuan provides a multi-dimensional vector-like type with dynamic sizes to model
the data processing in dynamic vector architectures. Also, a revised subset of
operations in the vector dialect is provided to make the high-level operations
suitable for being lowered to the target. Check the tablegen file for more
information.

Additionally, an VP dialect is also included in this project to enable the 
progressive lowering to the LLVM VP intrinsics. Some intrinsics are not yet
ported into the LLVM dialect, so `call_intrinsic` is used to represent them.

An experimental conversion pattern from Linalg dialect is implemented, which
is currently capable of converting a number of linalg operations to the Zuan
dialect, and later to RVV or LLVM VP intrinsics. This is similar to the existing
vectorization pattern of linalg dialect to the vector dialect, but the generated
code is quite different. See the benchmarks folder for more details.

Note that this is a prototype project, and the implementation is not yet
complete. The conversion pattern is still under development, and the lowering
pipeline is not yet fully finished.

## Roadmap

- [ ] Refine the conversion pattern from Linalg to Zuan
- [ ] More operations in the Zuan dialect if needed
- [ ] A more general lowering policy from Zuan to VP.

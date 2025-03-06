# Zuan MLIR

🚧 Work in progress

Zuan (纂, /tswan/) is an MLIR-based compiler framework designed for dynamic
sized vector architectures.

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

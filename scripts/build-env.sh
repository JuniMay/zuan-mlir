export ZUAN_SOURCE_DIR=${HOME}/zuan-mlir

export HOST_LLVM_BUILD_DIR=${ZUAN_SOURCE_DIR}/llvm-project/build
export HOST_ZUAN_BUILD_DIR=${ZUAN_SOURCE_DIR}/build

export RISCV_GNU_TOOLCHAIN_DIR=${HOST_ZUAN_BUILD_DIR}/third_party/riscv-gnu-toolchain
export RISCV_GNU_TOOLCHAIN_SYSROOT_DIR=${RISCV_GNU_TOOLCHAIN_DIR}/sysroot/

export RISCV_LLVM_BUILD_DIR=${ZUAN_SOURCE_DIR}/llvm-project/build-riscv
export RISCV_ZUAN_BUILD_DIR=${ZUAN_SOURCE_DIR}/build-riscv

export QEMU_LD_PREFIX=${RISCV_GNU_TOOLCHAIN_SYSROOT_DIR}
# Benchmark correctness for the fixed-width RVV baseline kernels assumes the
# old 256-bit vector-length configuration.
export QEMU_CPU=${QEMU_CPU:-max,v=true,vlen=256}

export TRITON_SOURCE_DIR=${ZUAN_SOURCE_DIR}/third_party/triton
export TRITON_CPU_SOURCE_DIR=${ZUAN_SOURCE_DIR}/third_party/triton-cpu
export TRITON_SHARED_SOURCE_DIR=${ZUAN_SOURCE_DIR}/third_party/triton_shared

export TRITON_BUILD_ROOT=${RISCV_ZUAN_BUILD_DIR}/third_party/triton
export ZUAN_TRITON_PYTHON_EXECUTABLE=${TRITON_BUILD_ROOT}/shared/venv/bin/python
export ZUAN_TRITON_SHARED_OPT=${TRITON_BUILD_ROOT}/shared/build/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt
export ZUAN_TRITON_LLVM_BINARY_DIR=${TRITON_BUILD_ROOT}/shared/llvm/bin
export ZUAN_TRITON_CPU_OPT=${TRITON_BUILD_ROOT}/cpu/build/bin/triton-opt
export ZUAN_TRITON_CPU_LLVM_BINARY_DIR=${TRITON_BUILD_ROOT}/cpu/llvm/bin

export TRITON_CPU_OPT=${ZUAN_TRITON_CPU_OPT}
export TRITON_SHARED_OPT=${ZUAN_TRITON_SHARED_OPT}

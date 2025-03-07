export ZUAN_SOURCE_DIR=${HOME}/zuan-mlir

export HOST_LLVM_BUILD_DIR=${ZUAN_SOURCE_DIR}/llvm-project/build
export HOST_ZUAN_BUILD_DIR=${ZUAN_SOURCE_DIR}/build

export RISCV_GNU_TOOLCHAIN_DIR=${HOST_ZUAN_BUILD_DIR}/third_party/riscv-gnu-toolchain
export RISCV_GNU_TOOLCHAIN_SYSROOT_DIR=${RISCV_GNU_TOOLCHAIN_DIR}/sysroot/

export RISCV_LLVM_BUILD_DIR=${ZUAN_SOURCE_DIR}/llvm-project/build-riscv
export RISCV_ZUAN_BUILD_DIR=${ZUAN_SOURCE_DIR}/build-riscv

export QEMU_LD_PREFIX=${RISCV_GNU_TOOLCHAIN_SYSROOT_DIR}

export TRITON_CPU_DIR=${HOME}/triton-cpu
export TRITON_SHARED_DIR=${HOME}/triton_shared

export TRITON_CPU_OPT=${TRITON_CPU_DIR}/triton-opt
export TRITON_SHARED_OPT=${TRITON_SHARED_DIR}/triton-shared-opt

# For triton-shared
export LLVM_BINARY_DIR=${HOST_LLVM_BUILD_DIR}/bin
export TRITON_SHARED_OPT_PATH=${TRITON_SHARED_OPT}

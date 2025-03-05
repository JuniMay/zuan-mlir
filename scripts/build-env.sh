export ZUAN_SOURCE_DIR=${HOME}/zuan-mlir

export HOST_LLVM_BUILD_DIR=${ZUAN_SOURCE_DIR}/llvm-project/build
export HOST_ZUAN_BUILD_DIR=${ZUAN_SOURCE_DIR}/build

export RISCV_GNU_TOOLCHAIN_DIR=${HOST_ZUAN_BUILD_DIR}/third_party/riscv-gnu-toolchain
export RISCV_GNU_TOOLCHAIN_SYSROOT_DIR=${RISCV_GNU_TOOLCHAIN_DIR}/sysroot/

export RISCV_LLVM_BUILD_DIR=${ZUAN_SOURCE_DIR}/llvm-project/build-riscv
export RISCV_MLIR_BUILD_DIR=${ZUAN_SOURCE_DIR}/llvm-project/build-riscv-mlir
export RISCV_ZUAN_BUILD_DIR=${ZUAN_SOURCE_DIR}/build-riscv

export QEMU_LD_PREFIX=${RISCV_GNU_TOOLCHAIN_SYSROOT_DIR}

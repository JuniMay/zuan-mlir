
mkdir -p ${RISCV_ZUAN_BUILD_DIR}
cd ${RISCV_ZUAN_BUILD_DIR}

PATH=${RISCV_GNU_TOOLCHAIN_DIR}/bin:${PATH} \
cmake -G Ninja .. \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
    -DMLIR_DIR=${RISCV_LLVM_BUILD_DIR}/lib/cmake/mlir \
    -DLLVM_DIR=${RISCV_LLVM_BUILD_DIR}/lib/cmake/llvm \
    -DCMAKE_CROSSCOMPILING=True \
    -DCMAKE_CROSSCOMPILING_EMULATOR=qemu-riscv64 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_NATIVE_ARCH=RISCV \
    -DLLVM_HOST_TRIPLE=riscv64-unknown-linux-gnu \
    -DCMAKE_C_COMPILER=${HOST_LLVM_BUILD_DIR}/bin/clang \
    -DCMAKE_CXX_COMPILER=${HOST_LLVM_BUILD_DIR}/bin/clang++ \
    -DCMAKE_C_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT_DIR} --gcc-toolchain=${RISCV_GNU_TOOLCHAIN_DIR}" \
    -DCMAKE_CXX_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT_DIR} --gcc-toolchain=${RISCV_GNU_TOOLCHAIN_DIR}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DENABLE_BENCHMARKS=ON \
    -DTRITON_CPU_OPT=${TRITON_CPU_OPT} \
    -DTRITON_SHARED_OPT=${TRITON_SHARED_OPT} \
    -DPython3_EXECUTABLE=$(which python3)

ninja

# Cross Compiled MLIR
mkdir ${RISCV_LLVM_BUILD_DIR}
cd ${RISCV_LLVM_BUILD_DIR}
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="RISCV" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DCMAKE_CROSSCOMPILING=True \
    -DLLVM_TARGET_ARCH=RISCV64 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_NATIVE_ARCH=RISCV \
    -DLLVM_HOST_TRIPLE=riscv64-unknown-linux-gnu \
    -DLLVM_DEFAULT_TARGET_TRIPLE=riscv64-unknown-linux-gnu \
    -DCMAKE_C_COMPILER=${HOST_LLVM_BUILD_DIR}/bin/clang \
    -DCMAKE_CXX_COMPILER=${HOST_LLVM_BUILD_DIR}/bin/clang++ \
    -DCMAKE_C_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT_DIR} --gcc-toolchain=${RISCV_GNU_TOOLCHAIN_DIR}" \
    -DCMAKE_CXX_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT_DIR} --gcc-toolchain=${RISCV_GNU_TOOLCHAIN_DIR}" \
    -DMLIR_TABLEGEN=${HOST_LLVM_BUILD_DIR}/bin/mlir-tblgen \
    -DLLVM_TABLEGEN=${HOST_LLVM_BUILD_DIR}/bin/llvm-tblgen \
    -DMLIR_LINALG_ODS_YAML_GEN=${HOST_LLVM_BUILD_DIR}/bin/mlir-linalg-ods-yaml-gen \
    -DMLIR_PDLL_TABLEGEN=${HOST_LLVM_BUILD_DIR}/bin/mlir-pdll \
    -DLLVM_ENABLE_ZSTD=Off

ninja

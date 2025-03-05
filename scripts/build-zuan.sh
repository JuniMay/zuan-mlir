mkdir build
cd build
cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm-project/build/lib/cmake/llvm \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DENABLE_RISCV_GNU_TOOLCHAIN=ON \
    -DENABLE_BENCHMARKS=ON

DYNO_BUILD_DIR=${DYNO_BUILD_DIR:-./build}
DYNO_LLVM_TOOLS_DIR=${DYNO_LLVM_TOOLS_DIR:-./llvm-project/build/bin}

${DYNO_BUILD_DIR}/bin/dyno-opt -convert-linalg-to-dyno -lower-dyno='target-rank=2' -dyno-stripmining="vf=4 scalable=true" -convert-dyno-to-vp="vf=4 scalable=true" -convert-vp-to-llvm -canonicalize -cse ./test/triton-shared/complex-gather-scatter.mlir -o test.mlir -debug-only=linalg-to-dyno,dyno-to-vp,dyno-unrolling,greedy-rewriter,dialect-conversion --mlir-print-ir-after-failure 2> test.txt
${DYNO_BUILD_DIR}/bin/dyno-opt ./test.mlir -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-vector-to-llvm -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts -canonicalize -cse -o test-llvm.mlir
${DYNO_BUILD_DIR}/bin/dyno-translate test-llvm.mlir -dyno-to-llvmir -o ./test.ll
${DYNO_LLVM_TOOLS_DIR}/opt -S -O3 ./test.ll -o test-O3.ll
${DYNO_LLVM_TOOLS_DIR}/llc -O3 -march=riscv64 -mattr=+m,+d,+c,+v,+zvfh test-O3.ll -o test-O3.s

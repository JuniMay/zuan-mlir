./build/bin/zuan-opt -convert-linalg-to-zuan -lower-zuan='target-rank=2' -zuan-stripmining="vf=4 scalable=true" -convert-zuan-to-vp="vf=4 scalable=true" -convert-vp-to-llvm -canonicalize -cse ./test/triton-shared/complex-gather-scatter.mlir -o test.mlir -debug-only=linalg-to-zuan,zuan-to-vp,zuan-unrolling,greedy-rewriter,dialect-conversion --mlir-print-ir-after-failure 2> test.txt
./build/bin/zuan-opt ./test.mlir -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-vector-to-llvm -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts -canonicalize -cse -o test-llvm.mlir
./build/bin/zuan-translate test-llvm.mlir -zuan-to-llvmir -o ./test.ll
./llvm-project/build/bin/opt -S -O3 ./test.ll -o test-O3.ll
./llvm-project/build/bin/llc -O3 -march=riscv64 -mattr=+m,+d,+c,+v,+zvfh test-O3.ll -o test-O3.s

// RUN: dyno-opt %s \
// RUN:   -convert-linalg-to-dyno \
// RUN:   -lower-dyno='target-rank=2' \
// RUN:   -dyno-stripmining='vf=16 uf=2 scalable=true \
// RUN:                     reduction-mode=sequential \
// RUN:                     fp-policy=strict' \
// RUN: | tee %t.lowered.mlir \
// RUN: | FileCheck %s \
// RUN: && mv -f %t.lowered.mlir \
// RUN:   $(dirname %t)/linalg-reduce-ordered.lowered.mlir

func.func @kernel(%a: memref<?x?xf32>, %dst: memref<?xf32>) {
  linalg.reduce { arith.addf } ins(%a: memref<?x?xf32>) outs(%dst: memref<?xf32>) dimensions=[0]
  return
}

// CHECK-LABEL: func.func @kernel
// CHECK: scf.while
// CHECK: scf.for %{{.*}} = %c0 to %{{.*}} step %c1 iter_args(%{{.*}} = %{{.*}}) -> (!dyno.tile<?xf32>)
// CHECK: arith.addf %{{.*}}, %{{.*}} : !dyno.tile<?xf32>
// CHECK-NOT: dyno.reduction
// CHECK: dyno.store %{{.*}}, %{{.*}} {dyno.stripmined}

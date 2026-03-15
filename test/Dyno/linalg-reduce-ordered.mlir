// RUN: dyno-opt -convert-linalg-to-dyno -lower-dyno='target-rank=2' -dyno-stripmining='vf=16 uf=2 scalable=true reduction-mode=sequential fp-policy=strict' %s | FileCheck %s

func.func @kernel(%a: memref<?x?xf32>, %dst: memref<?xf32>) {
  linalg.reduce { arith.addf } ins(%a: memref<?x?xf32>) outs(%dst: memref<?xf32>) dimensions=[0]
  return
}

// CHECK-LABEL: func.func @kernel
// CHECK: scf.while
// CHECK: %[[ACC:.*]] = scf.for
// CHECK: arith.addf %{{.*}}, %{{.*}} : !dyno.tile<?xf32>
// CHECK: dyno.store %[[ACC]], %{{.*}} {dyno.stripmined}
// CHECK-NOT: dyno.reduction <add>

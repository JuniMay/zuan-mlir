// RUN: dyno-opt -convert-linalg-to-dyno -lower-dyno='target-rank=2' -dyno-stripmining='vf=16 uf=2 scalable=true reduction-mode=sequential fp-policy=strict' %s | FileCheck %s

func.func @kernel(%a: memref<?x?xf32>, %dst: memref<?xf32>) {
  linalg.reduce { arith.addf } ins(%a: memref<?x?xf32>) outs(%dst: memref<?xf32>) dimensions=[0]
  return
}

// CHECK-LABEL: func.func @kernel
// CHECK: scf.while
// CHECK: dyno.reduction <add> %{{.*}} [0], %{{.*}} : !dyno.tile<?x?xf32>, !dyno.tile<?xf32>
// CHECK: dyno.store %{{.*}}, %{{.*}} {dyno.stripmined}

// RUN: dyno-opt %s \
// RUN:   -lower-dyno='target-rank=2' \
// RUN:   -dyno-stripmining='vf=8 uf=2 scalable=true reduction-mode=sequential \
// RUN:                     fp-policy=strict' \
// RUN:   -convert-dyno-to-vp='vf=8 scalable=true' \
// RUN: | tee %t.vp.mlir \
// RUN: | FileCheck %s \
// RUN: && mv -f %t.vp.mlir \
// RUN:   $(dirname %t)/convert-to-vp.lowered.mlir

// CHECK-LABEL: func.func @masked_add
func.func @masked_add(%a: memref<?xf32>, %b: memref<?xf32>, %m: memref<?xi1>,
                      %dst: memref<?xf32>, %maskedoff: memref<?xf32>) {
  %a_tile = dyno.load %a : memref<?xf32>
  %b_tile = dyno.load %b : memref<?xf32>
  %m_tile = dyno.load %m : memref<?xi1>
  %maskedoff_tile = dyno.load %maskedoff : memref<?xf32>
  %res = dyno.mask %m_tile : !dyno.tile<?xi1>, %maskedoff_tile : !dyno.tile<?xf32> {
    %add = arith.addf %a_tile, %b_tile : !dyno.tile<?xf32>
    dyno.mask_yield %add : !dyno.tile<?xf32>
  } : !dyno.tile<?xf32>
  dyno.store %res, %dst : !dyno.tile<?xf32>, memref<?xf32>
  return
}

// CHECK: vp.predicate %{{.*}} : index, mask = %{{.*}} : vector<[8]xi1>, passthru = none, maskedoff = %{{.*}} : vector<[8]xf32> {
// CHECK: arith.addf

func.func @sequential_reduce(%src: memref<?xf32>) -> f32 {
  %tile = dyno.load %src : memref<?xf32>
  %reduced = dyno.reduction <add> %tile [0] : !dyno.tile<?xf32>
  %scalar = dyno.extract %reduced : !dyno.tile<f32>
  return %scalar : f32
}

// CHECK-LABEL: func.func @sequential_reduce
// CHECK: scf.while
// CHECK: vp.predicate %{{.*}} : index, mask = none, passthru = none, maskedoff = none {
// CHECK: vector.reduction <add>, %{{.*}}, %{{.*}} : vector<[8]xf32> into f32
// CHECK-NOT: fastmath<reassoc>
// CHECK-NOT: scf.for
// CHECK-NOT: memref.load
func.func @rank2_store(%src: memref<2x?xf32>, %dst: memref<2x?xf32>) {
  %tile = dyno.load %src : memref<2x?xf32>
  dyno.store %tile, %dst : !dyno.tile<2x?xf32>, memref<2x?xf32>
  return
}

// CHECK-LABEL: func.func @rank2_store
// CHECK: scf.for
// CHECK: scf.for
// CHECK: vp.load
// CHECK: vp.store

// RUN: dyno-opt -lower-dyno='target-rank=2' -dyno-stripmining='vf=8 scalable=true' -convert-dyno-to-vp='vf=8 scalable=true' %s | FileCheck %s

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

// CHECK-LABEL: func.func @row_pack_store
// CHECK: scf.for
// CHECK: vp.load
// CHECK: vp.store
func.func @row_pack_store(%src: memref<2x?xf32>, %dst: memref<2x?xf32>) {
  %tile = dyno.load %src : memref<2x?xf32>
  dyno.store %tile, %dst : !dyno.tile<2x?xf32>, memref<2x?xf32>
  return
}

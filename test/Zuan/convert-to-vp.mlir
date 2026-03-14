// RUN: zuan-opt -lower-zuan='target-rank=2' -zuan-stripmining='vf=8 scalable=true' -convert-zuan-to-vp='vf=8 scalable=true' %s | FileCheck %s

// CHECK-LABEL: func.func @masked_add
func.func @masked_add(%a: memref<?xf32>, %b: memref<?xf32>, %m: memref<?xi1>,
                      %dst: memref<?xf32>, %maskedoff: memref<?xf32>) {
  %a_tile = zuan.load %a : memref<?xf32>
  %b_tile = zuan.load %b : memref<?xf32>
  %m_tile = zuan.load %m : memref<?xi1>
  %maskedoff_tile = zuan.load %maskedoff : memref<?xf32>
  %res = zuan.mask %m_tile : !zuan.tile<?xi1>, %maskedoff_tile : !zuan.tile<?xf32> {
    %add = arith.addf %a_tile, %b_tile : !zuan.tile<?xf32>
    zuan.mask_yield %add : !zuan.tile<?xf32>
  } : !zuan.tile<?xf32>
  zuan.store %res, %dst : !zuan.tile<?xf32>, memref<?xf32>
  return
}

// CHECK: vp.predicate %{{.*}} : index, mask = %{{.*}} : vector<[8]xi1>, passthru = none, maskedoff = %{{.*}} : vector<[8]xf32> {
// CHECK: arith.addf

// CHECK-LABEL: func.func @outer_row_scalarization
// CHECK: scf.for
// CHECK: memref.load %{{.*}}[] : memref<f32
// CHECK: vector.broadcast
// CHECK-NOT: vector.extract
func.func @outer_row_scalarization(%lhs: memref<2xf32>, %rhs: memref<?xf32>,
                                   %dst: memref<2x?xf32>) {
  %lhs_tile = zuan.load %lhs : memref<2xf32>
  %rhs_tile = zuan.load %rhs : memref<?xf32>
  %res = zuan.outer <mul> %lhs_tile, %rhs_tile : !zuan.tile<2xf32>, !zuan.tile<?xf32>
  zuan.store %res, %dst : !zuan.tile<2x?xf32>, memref<2x?xf32>
  return
}

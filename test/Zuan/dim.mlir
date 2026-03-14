// RUN: zuan-opt -resolve-zuan-dims %s | FileCheck %s --check-prefix=RESOLVE
// RUN: zuan-opt -convert-zuan-to-vp='vf=8 scalable=true' %s | FileCheck %s --check-prefix=VP

// RESOLVE-LABEL: func.func @dim_static
func.func @dim_static(%arg0: !zuan.tile<4x?xf32>) -> index {
  %d0 = zuan.dim %arg0, 0 : !zuan.tile<4x?xf32>
  // RESOLVE: arith.constant 4 : index
  return %d0 : index
}

// RESOLVE-LABEL: func.func @dim_load
func.func @dim_load(%m: memref<?x8xf32>) -> (index, index) {
  %tile = zuan.load %m : memref<?x8xf32>
  %d0 = zuan.dim %tile, 0 : !zuan.tile<?x8xf32>
  %d1 = zuan.dim %tile, 1 : !zuan.tile<?x8xf32>
  // RESOLVE: %[[C8:.*]] = arith.constant 8 : index
  // RESOLVE: memref.dim %{{.*}}, %{{.*}} : memref<?x8xf32>
  return %d0, %d1 : index, index
}

// RESOLVE-LABEL: func.func @dim_forwarding
func.func @dim_forwarding(%a: memref<?x?xf32>, %b: memref<?x?xf32>,
                          %m: memref<?x?xi1>) -> (index, index, index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %a_tile = zuan.load %a : memref<?x?xf32>
  %b_tile = zuan.load %b : memref<?x?xf32>
  %m_tile = zuan.load %m : memref<?x?xi1>
  %a0 = memref.dim %a, %c0 : memref<?x?xf32>
  %a1 = memref.dim %a, %c1 : memref<?x?xf32>
  %sum = arith.addf %a_tile, %b_tile : !zuan.tile<?x?xf32>
  %cmp = arith.cmpf olt, %a_tile, %b_tile : !zuan.tile<?x?xf32>
  %step = zuan.step %c0, 0, [%a0, %a1] : index
  %select = zuan.select %cmp, %sum, %a_tile :
      !zuan.tile<?x?xi1>, !zuan.tile<?x?xf32>
  %masked = zuan.mask %m_tile : !zuan.tile<?x?xi1> {
    zuan.mask_yield %select : !zuan.tile<?x?xf32>
  } : !zuan.tile<?x?xf32>
  %d0 = zuan.dim %masked, 0 : !zuan.tile<?x?xf32>
  %d1 = zuan.dim %masked, 1 : !zuan.tile<?x?xf32>
  %s0 = zuan.dim %step, 0 : !zuan.tile<?x?xindex>
  %s1 = zuan.dim %step, 1 : !zuan.tile<?x?xindex>
  // RESOLVE-NOT: zuan.dim
  return %d0, %d1, %s0, %s1 : index, index, index, index
}

// RESOLVE-LABEL: func.func @dim_matmul_and_reduction
func.func @dim_matmul_and_reduction(%lhs: memref<?x?xf32>, %rhs: memref<?x?xf32>,
                                    %src: memref<?x?xf32>) -> (index, index, index) {
  %lhs_tile = zuan.load %lhs : memref<?x?xf32>
  %rhs_tile = zuan.load %rhs : memref<?x?xf32>
  %src_tile = zuan.load %src : memref<?x?xf32>
  %mm = zuan.matmul %lhs_tile, %rhs_tile : !zuan.tile<?x?xf32>, !zuan.tile<?x?xf32>
  %red = zuan.reduction <add> %src_tile [1] : !zuan.tile<?x?xf32>
  %m = zuan.dim %mm, 0 : !zuan.tile<?x?xf32>
  %n = zuan.dim %mm, 1 : !zuan.tile<?x?xf32>
  %r = zuan.dim %red, 0 : !zuan.tile<?xf32>
  // RESOLVE-NOT: zuan.dim
  return %m, %n, %r : index, index, index
}

// VP-LABEL: func.func @vp_resolve_load_dim_before_erase
func.func @vp_resolve_load_dim_before_erase(%a: memref<?xf32>, %b: memref<?xf32>) -> index {
  %tile = zuan.load %a : memref<?xf32>
  %dim = zuan.dim %tile, 0 : !zuan.tile<?xf32>
  zuan.store %tile, %b : !zuan.tile<?xf32>, memref<?xf32>
  // VP-NOT: zuan.dim
  // VP: memref.dim %a
  return %dim : index
}

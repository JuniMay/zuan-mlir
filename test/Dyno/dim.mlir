// RUN: dyno-opt -resolve-dyno-dims %s | FileCheck %s --check-prefix=RESOLVE
// RUN: dyno-opt -convert-dyno-to-vp='vf=8 scalable=true' %s | FileCheck %s --check-prefix=VP

// RESOLVE-LABEL: func.func @dim_static
func.func @dim_static(%arg0: !dyno.tile<4x?xf32>) -> index {
  %d0 = dyno.dim %arg0, 0 : !dyno.tile<4x?xf32>
  // RESOLVE: arith.constant 4 : index
  return %d0 : index
}

// RESOLVE-LABEL: func.func @dim_load
func.func @dim_load(%m: memref<?x8xf32>) -> (index, index) {
  %tile = dyno.load %m : memref<?x8xf32>
  %d0 = dyno.dim %tile, 0 : !dyno.tile<?x8xf32>
  %d1 = dyno.dim %tile, 1 : !dyno.tile<?x8xf32>
  // RESOLVE: %[[C8:.*]] = arith.constant 8 : index
  // RESOLVE: memref.dim %{{.*}}, %{{.*}} : memref<?x8xf32>
  return %d0, %d1 : index, index
}

// RESOLVE-LABEL: func.func @dim_forwarding
func.func @dim_forwarding(%a: memref<?x?xf32>, %b: memref<?x?xf32>,
                          %m: memref<?x?xi1>) -> (index, index, index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %a_tile = dyno.load %a : memref<?x?xf32>
  %b_tile = dyno.load %b : memref<?x?xf32>
  %m_tile = dyno.load %m : memref<?x?xi1>
  %a0 = memref.dim %a, %c0 : memref<?x?xf32>
  %a1 = memref.dim %a, %c1 : memref<?x?xf32>
  %sum = arith.addf %a_tile, %b_tile : !dyno.tile<?x?xf32>
  %cmp = arith.cmpf olt, %a_tile, %b_tile : !dyno.tile<?x?xf32>
  %step = dyno.step %c0, 0, [%a0, %a1] : index
  %select = dyno.select %cmp, %sum, %a_tile :
      !dyno.tile<?x?xi1>, !dyno.tile<?x?xf32>
  %masked = dyno.mask %m_tile : !dyno.tile<?x?xi1> {
    dyno.mask_yield %select : !dyno.tile<?x?xf32>
  } : !dyno.tile<?x?xf32>
  %d0 = dyno.dim %masked, 0 : !dyno.tile<?x?xf32>
  %d1 = dyno.dim %masked, 1 : !dyno.tile<?x?xf32>
  %s0 = dyno.dim %step, 0 : !dyno.tile<?x?xindex>
  %s1 = dyno.dim %step, 1 : !dyno.tile<?x?xindex>
  // RESOLVE-NOT: dyno.dim
  return %d0, %d1, %s0, %s1 : index, index, index, index
}

// RESOLVE-LABEL: func.func @dim_reduction_and_splat
func.func @dim_reduction_and_splat(%src: memref<?x?x?xf32>,
                                   %init: memref<?x?xf32>) -> (index, index, index) {
  %src_tile = dyno.load %src : memref<?x?x?xf32>
  %init_tile = dyno.load %init : memref<?x?xf32>
  %red = dyno.reduction <add> %src_tile [1], %init_tile :
      !dyno.tile<?x?x?xf32>, !dyno.tile<?x?xf32>
  %red0 = dyno.dim %red, 0 : !dyno.tile<?x?xf32>
  %red1 = dyno.dim %red, 1 : !dyno.tile<?x?xf32>
  %splat = dyno.splat %red [3] : !dyno.tile<?x?xf32>
  %sp0 = dyno.dim %splat, 0 : !dyno.tile<3x?x?xf32>
  // RESOLVE-NOT: dyno.dim
  return %red0, %red1, %sp0 : index, index, index
}

// VP-LABEL: func.func @vp_resolve_load_dim_before_erase
func.func @vp_resolve_load_dim_before_erase(%a: memref<?xf32>, %b: memref<?xf32>) -> index {
  %tile = dyno.load %a : memref<?xf32>
  %dim = dyno.dim %tile, 0 : !dyno.tile<?xf32>
  dyno.store %tile, %b : !dyno.tile<?xf32>, memref<?xf32>
  // VP-NOT: dyno.dim
  // VP: memref.dim %a
  return %dim : index
}

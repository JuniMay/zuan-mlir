// RUN: dyno-opt -dyno-strength-reduction %s | FileCheck %s

// CHECK-LABEL: func.func @splatcast
func.func @splatcast(%arg0: index, %arg1: index, %arg2: memref<1x?xi32>) {
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg2, %c1 : memref<1x?xi32>
  // CHECK:      arith.index_cast
  // CHECK-NEXT: dyno.splat
  %splat = dyno.splat %arg0 [1, %dim] : index
  %cast = dyno.cast <indexcast> %splat :
      !dyno.tile<1x?xindex> to !dyno.tile<1x?xi32>
  dyno.store %cast, %arg2 : !dyno.tile<1x?xi32>, memref<1x?xi32>
  return
}

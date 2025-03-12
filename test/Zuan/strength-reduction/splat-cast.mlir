// RUN: zuan-opt -zuan-strength-reduction %s | FileCheck %s

// CHECK-LABEL: func.func @splatcast
func.func @splatcast(%arg0: index, %arg1: index, %arg2: memref<1x?xi32>) {
  %1 = arith.constant 1 : index
  %dim = memref.dim %arg2, %1 : memref<1x?xi32>

  zuan.dynamic (%arg2 : memref<1x?xi32>) {
  ^bb0(%arg2_tile: !zuan.tile<1x?xindex>):
    // CHECK:      arith.index_cast
    // CHECK-NEXT: zuan.splat
    %splat = zuan.splat %arg0 [1, %dim] : index
    %cast = zuan.cast <indexcast> %splat : !zuan.tile<1x?xindex> to !zuan.tile<1x?xi32>
    zuan.yield {
      zuan.store %cast, %arg2 : !zuan.tile<1x?xi32>, memref<1x?xi32>
    }
  }
  return
}

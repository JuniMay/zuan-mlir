// RUN: zuan-opt -zuan-strength-reduction %s | FileCheck %s

// CHECK-LABEL: func.func @stepunit
func.func @stepunit(%arg0: index, %size: index, %arg3: memref<1x?xindex>) {
  zuan.dynamic (%arg3 : memref<1x?xindex>) {
  ^bb0(%arg3_tile: !zuan.tile<1x?xindex>):
    // CHECK: zuan.splat
    %step = zuan.step %arg0, 0, [1, %size] : index
    zuan.yield {
      zuan.store %step, %arg3 : !zuan.tile<1x?xindex>, memref<1x?xindex>
    }
  }
  return
}

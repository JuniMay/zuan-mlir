// RUN: dyno-opt -dyno-strength-reduction %s | FileCheck %s

// CHECK-LABEL: func.func @stepunit
func.func @stepunit(%arg0: index, %size: index, %arg3: memref<1x?xindex>) {
  // CHECK: dyno.splat
  %step = dyno.step %arg0, 0, [1, %size] : index
  dyno.store %step, %arg3 : !dyno.tile<1x?xindex>, memref<1x?xindex>
  return
}

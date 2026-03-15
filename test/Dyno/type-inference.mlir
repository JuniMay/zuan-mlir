// RUN: dyno-opt %s | FileCheck %s

// CHECK-LABEL: func.func @load_and_reduction_inference
func.func @load_and_reduction_inference(%src: memref<4x?xf32>) -> index {
  %tile = dyno.load %src : memref<4x?xf32>
  %red = dyno.reduction <add> %tile [1] : !dyno.tile<4x?xf32>
  // CHECK: dyno.dim %{{.*}}, 0 : !dyno.tile<4xf32>
  %d0 = dyno.dim %red, 0 : !dyno.tile<4xf32>
  return %d0 : index
}

// CHECK-LABEL: func.func @splat_and_step_inference
func.func @splat_and_step_inference(%arg0: !dyno.tile<2xf32>,
                                    %start: index) -> (index, index) {
  %splat = dyno.splat %arg0 [3] : !dyno.tile<2xf32>
  %step = dyno.step %start, 1, [2, 4] : index
  // CHECK: dyno.dim %{{.*}}, 0 : !dyno.tile<3x2xf32>
  %d0 = dyno.dim %splat, 0 : !dyno.tile<3x2xf32>
  // CHECK: dyno.dim %{{.*}}, 1 : !dyno.tile<2x4xindex>
  %d1 = dyno.dim %step, 1 : !dyno.tile<2x4xindex>
  return %d0, %d1 : index, index
}

// CHECK-LABEL: func.func @gather_inference
func.func @gather_inference(%src: memref<?x?xf32>, %idx0: !dyno.tile<?xindex>,
                            %idx1: !dyno.tile<?xindex>) -> index {
  %gather = dyno.gather %src[%idx0, %idx1] :
      memref<?x?xf32>, !dyno.tile<?xindex>, !dyno.tile<?xindex>
  // CHECK: dyno.dim %{{.*}}, 0 : !dyno.tile<?xf32>
  %d0 = dyno.dim %gather, 0 : !dyno.tile<?xf32>
  return %d0 : index
}

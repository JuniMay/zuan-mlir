// RUN: dyno-opt %s | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: dyno-opt %s -resolve-dyno-dims | FileCheck %s --check-prefix=RESOLVE

// ROUNDTRIP-LABEL: func.func @inference_and_dim
func.func @inference_and_dim(%acc: memref<8xf32>, %chunk: memref<?xf32>) -> index {
  %acc_tile = dyno.load %acc : memref<8xf32>
  %chunk_tile = dyno.load %chunk : memref<?xf32>
  %res = dyno.reduction_accumulate <add> %acc_tile, %chunk_tile :
      !dyno.tile<8xf32>, !dyno.tile<?xf32>
  // ROUNDTRIP: dyno.reduction_accumulate <add> %{{.*}}, %{{.*}} : !dyno.tile<8xf32>, !dyno.tile<?xf32>
  // ROUNDTRIP: dyno.dim %{{.*}}, 0 : !dyno.tile<8xf32>
  %d0 = dyno.dim %res, 0 : !dyno.tile<8xf32>
  return %d0 : index
}

// RESOLVE-LABEL: func.func @inference_and_dim
// RESOLVE-NOT: dyno.dim
// RESOLVE: arith.constant 8 : index

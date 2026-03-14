// RUN: dyno-opt %s | FileCheck %s

// CHECK-LABEL: func.func @flat_roundtrip
func.func @flat_roundtrip(%a: memref<?x?xf32>, %b: memref<?x?xf32>,
                          %c: memref<?x?xf32>) {
  %a_tile = dyno.load %a : memref<?x?xf32>
  %b_tile = dyno.load %b : memref<?x?xf32>
  %c_tile = dyno.load %c : memref<?x?xf32>
  // CHECK: %[[MM:.*]] = dyno.matmul %{{.*}}, %{{.*}}
  %mm = dyno.matmul %a_tile, %b_tile : !dyno.tile<?x?xf32>, !dyno.tile<?x?xf32>
  %sum = arith.addf %mm, %c_tile : !dyno.tile<?x?xf32>
  // CHECK: dyno.store %{{.*}}, %{{.*}} : <?x?xf32>, memref<?x?xf32>
  dyno.store %sum, %c : !dyno.tile<?x?xf32>, memref<?x?xf32>
  return
}

// CHECK-LABEL: func.func @scalar_roundtrip
func.func @scalar_roundtrip(%arg0: f32, %dst: memref<?xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %n = memref.dim %dst, %c0 : memref<?xf32>
  %tile = dyno.splat %arg0 [%n] : f32
  %reduced = dyno.reduction <add> %tile [0] : !dyno.tile<?xf32>
  // CHECK: %[[SCALAR:.*]] = dyno.extract %{{.*}} : !dyno.tile<f32>
  %scalar = dyno.extract %reduced : !dyno.tile<f32>
  return %scalar : f32
}

// CHECK-LABEL: func.func @mask_and_gather
func.func @mask_and_gather(%src: memref<?x?xf32>, %idx0: memref<?xindex>,
                           %idx1: memref<?xindex>, %mask: memref<?xi1>,
                           %dst: memref<?xf32>) {
  %idx0_tile = dyno.load %idx0 : memref<?xindex>
  %idx1_tile = dyno.load %idx1 : memref<?xindex>
  %mask_tile = dyno.load %mask : memref<?xi1>
  %gathered = dyno.gather %src[%idx0_tile, %idx1_tile] :
      memref<?x?xf32>, !dyno.tile<?xindex>, !dyno.tile<?xindex>
  %masked = dyno.mask %mask_tile : !dyno.tile<?xi1> {
    dyno.mask_yield %gathered : !dyno.tile<?xf32>
  } : !dyno.tile<?xf32>
  // CHECK: dyno.scatter %{{.*}}, %{{.*}}[%{{.*}}]
  dyno.scatter %masked, %dst[%idx0_tile] :
      !dyno.tile<?xf32>, memref<?xf32>, !dyno.tile<?xindex>
  return
}

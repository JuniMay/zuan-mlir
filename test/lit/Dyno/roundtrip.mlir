// RUN: dyno-opt %s | FileCheck %s

// CHECK-LABEL: func.func @core_roundtrip
func.func @core_roundtrip(%src: memref<?x?x?xf32>, %init: memref<?x?xf32>,
                          %dst: memref<?x?xf32>) {
  %src_tile = dyno.load %src : memref<?x?x?xf32>
  %init_tile = dyno.load %init : memref<?x?xf32>
  %mul = arith.mulf %src_tile, %src_tile : !dyno.tile<?x?x?xf32>
  // CHECK: dyno.reduction <add> %{{.*}} [1], %{{.*}} : !dyno.tile<?x?x?xf32>, !dyno.tile<?x?xf32>
  %red = dyno.reduction <add> %mul [1], %init_tile :
      !dyno.tile<?x?x?xf32>, !dyno.tile<?x?xf32>
  // CHECK: dyno.store %{{.*}}, %{{.*}} : <?x?xf32>, memref<?x?xf32>
  dyno.store %red, %dst : !dyno.tile<?x?xf32>, memref<?x?xf32>
  return
}

// CHECK-LABEL: func.func @scalar_and_mask_roundtrip
func.func @scalar_and_mask_roundtrip(%arg0: f32, %mask: memref<?xi1>,
                                     %dst: memref<?xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %n = memref.dim %dst, %c0 : memref<?xf32>
  %mask_tile = dyno.load %mask : memref<?xi1>
  %tile = dyno.splat %arg0 [%n] : f32
  %masked = dyno.mask %mask_tile : !dyno.tile<?xi1> {
    dyno.mask_yield %tile : !dyno.tile<?xf32>
  } : !dyno.tile<?xf32>
  %reduced = dyno.reduction <add> %masked [0] : !dyno.tile<?xf32>
  // CHECK: dyno.extract %{{.*}} : !dyno.tile<f32>
  %scalar = dyno.extract %reduced : !dyno.tile<f32>
  return %scalar : f32
}

// CHECK-LABEL: func.func @gather_scatter_roundtrip
func.func @gather_scatter_roundtrip(%src: memref<?x?xf32>,
                                    %idx0: memref<?xindex>,
                                    %idx1: memref<?xindex>,
                                    %dst: memref<?xf32>) {
  %idx0_tile = dyno.load %idx0 : memref<?xindex>
  %idx1_tile = dyno.load %idx1 : memref<?xindex>
  %gathered = dyno.gather %src[%idx0_tile, %idx1_tile] :
      memref<?x?xf32>, !dyno.tile<?xindex>, !dyno.tile<?xindex>
  // CHECK: dyno.scatter %{{.*}}, %{{.*}}[%{{.*}}] : <?xf32>, memref<?xf32>, !dyno.tile<?xindex>
  dyno.scatter %gathered, %dst[%idx0_tile] :
      !dyno.tile<?xf32>, memref<?xf32>, !dyno.tile<?xindex>
  return
}

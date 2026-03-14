// RUN: zuan-opt %s | FileCheck %s

// CHECK-LABEL: func.func @flat_roundtrip
func.func @flat_roundtrip(%a: memref<?x?xf32>, %b: memref<?x?xf32>,
                          %c: memref<?x?xf32>) {
  %a_tile = zuan.load %a : memref<?x?xf32>
  %b_tile = zuan.load %b : memref<?x?xf32>
  %c_tile = zuan.load %c : memref<?x?xf32>
  // CHECK: %[[MM:.*]] = zuan.matmul %{{.*}}, %{{.*}}
  %mm = zuan.matmul %a_tile, %b_tile : !zuan.tile<?x?xf32>, !zuan.tile<?x?xf32>
  %sum = arith.addf %mm, %c_tile : !zuan.tile<?x?xf32>
  // CHECK: zuan.store %{{.*}}, %{{.*}} : <?x?xf32>, memref<?x?xf32>
  zuan.store %sum, %c : !zuan.tile<?x?xf32>, memref<?x?xf32>
  return
}

// CHECK-LABEL: func.func @scalar_roundtrip
func.func @scalar_roundtrip(%arg0: f32, %dst: memref<?xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %n = memref.dim %dst, %c0 : memref<?xf32>
  %tile = zuan.splat %arg0 [%n] : f32
  %reduced = zuan.reduction <add> %tile [0] : !zuan.tile<?xf32>
  // CHECK: %[[SCALAR:.*]] = zuan.extract %{{.*}} : !zuan.tile<f32>
  %scalar = zuan.extract %reduced : !zuan.tile<f32>
  return %scalar : f32
}

// CHECK-LABEL: func.func @mask_and_gather
func.func @mask_and_gather(%src: memref<?x?xf32>, %idx0: memref<?xindex>,
                           %idx1: memref<?xindex>, %mask: memref<?xi1>,
                           %dst: memref<?xf32>) {
  %idx0_tile = zuan.load %idx0 : memref<?xindex>
  %idx1_tile = zuan.load %idx1 : memref<?xindex>
  %mask_tile = zuan.load %mask : memref<?xi1>
  %gathered = zuan.gather %src[%idx0_tile, %idx1_tile] :
      memref<?x?xf32>, !zuan.tile<?xindex>, !zuan.tile<?xindex>
  %masked = zuan.mask %mask_tile : !zuan.tile<?xi1> {
    zuan.mask_yield %gathered : !zuan.tile<?xf32>
  } : !zuan.tile<?xf32>
  // CHECK: zuan.scatter %{{.*}}, %{{.*}}[%{{.*}}]
  zuan.scatter %masked, %dst[%idx0_tile] :
      !zuan.tile<?xf32>, memref<?xf32>, !zuan.tile<?xindex>
  return
}

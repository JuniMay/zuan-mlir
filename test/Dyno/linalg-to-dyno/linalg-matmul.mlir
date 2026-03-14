// RUN: dyno-opt -convert-linalg-to-dyno %s | FileCheck %s

// CHECK-LABEL: func.func @linalg_matmul
func.func @linalg_matmul(%a: memref<?x?xf32>, %b: memref<?x?xf32>,
                         %c: memref<?x?xf32>) {
  // CHECK:      %[[A:.*]] = dyno.load %{{.*}} : memref<?x?x?xf32, strided<[?, 0, 1]>>
  // CHECK-NEXT: %[[B:.*]] = dyno.load %{{.*}} : memref<?x?x?xf32, strided<[0, 1, ?]>>
  // CHECK-NEXT: %[[C:.*]] = dyno.load %{{.*}} : memref<?x?xf32, strided<[?, 1]>>
  // CHECK-NEXT: %[[MUL:.*]] = arith.mulf %[[A]], %[[B]] : !dyno.tile<?x?x?xf32>
  // CHECK-NEXT: %[[RED:.*]] = dyno.reduction <add> %[[MUL]] [2], %[[C]] : !dyno.tile<?x?x?xf32>, !dyno.tile<?x?xf32>
  // CHECK-NEXT: dyno.store %[[RED]], %{{.*}} : <?x?xf32>, memref<?x?xf32, strided<[?, 1]>>
  linalg.matmul ins(%a, %b : memref<?x?xf32>, memref<?x?xf32>) outs(%c: memref<?x?xf32>)
  return
}

// CHECK-LABEL: func.func @linalg_matmul_transpose_a
func.func @linalg_matmul_transpose_a(%a: memref<?x?xf32>, %b: memref<?x?xf32>,
                                     %c: memref<?x?xf32>) {
  // CHECK: dyno.load %{{.*}} : memref<?x?x?xf32, strided<[1, 0, ?]>>
  // CHECK: dyno.load %{{.*}} : memref<?x?x?xf32, strided<[0, 1, ?]>>
  // CHECK: dyno.store %{{.*}}, %{{.*}} : <?x?xf32>, memref<?x?xf32, strided<[?, 1]>>
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d2, d0)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                      ]
                      ins(%a, %b : memref<?x?xf32>, memref<?x?xf32>)
                      outs(%c: memref<?x?xf32>)
  return
}

// CHECK-LABEL: func.func @linalg_matmul_transpose_b
func.func @linalg_matmul_transpose_b(%a: memref<?x?xf32>, %b: memref<?x?xf32>,
                                     %c: memref<?x?xf32>) {
  // CHECK: dyno.load %{{.*}} : memref<?x?x?xf32, strided<[?, 0, 1]>>
  // CHECK: dyno.load %{{.*}} : memref<?x?x?xf32, strided<[0, ?, 1]>>
  // CHECK: dyno.store %{{.*}}, %{{.*}} : <?x?xf32>, memref<?x?xf32, strided<[?, 1]>>
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                      ]
                      ins(%a, %b : memref<?x?xf32>, memref<?x?xf32>)
                      outs(%c: memref<?x?xf32>)
  return
}

// RUN: zuan-opt -convert-linalg-to-zuan %s | FileCheck %s

// CHECK-LABEL: func.func @linalg_matmul
func.func @linalg_matmul(%a: memref<?x?xf32>, %b: memref<?x?xf32>,
                         %c: memref<?x?xf32>) {
  // CHECK:      %[[A:.*]] = zuan.load %{{.*}} : memref<?x?x?xf32, strided<[?, 0, 1]>>
  // CHECK-NEXT: %[[B:.*]] = zuan.load %{{.*}} : memref<?x?x?xf32, strided<[0, 1, ?]>>
  // CHECK-NEXT: %[[C:.*]] = zuan.load %{{.*}} : memref<?x?xf32, strided<[?, 1]>>
  // CHECK-NEXT: %[[MUL:.*]] = arith.mulf %[[A]], %[[B]] : !zuan.tile<?x?x?xf32>
  // CHECK-NEXT: %[[RED:.*]] = zuan.reduction <add> %[[MUL]] [2], %[[C]] : !zuan.tile<?x?x?xf32>, !zuan.tile<?x?xf32>
  // CHECK-NEXT: zuan.store %[[RED]], %{{.*}} : <?x?xf32>, memref<?x?xf32, strided<[?, 1]>>
  linalg.matmul ins(%a, %b : memref<?x?xf32>, memref<?x?xf32>) outs(%c: memref<?x?xf32>)
  return
}

// CHECK-LABEL: func.func @linalg_matmul_transpose_a
func.func @linalg_matmul_transpose_a(%a: memref<?x?xf32>, %b: memref<?x?xf32>,
                                     %c: memref<?x?xf32>) {
  // CHECK: zuan.load %{{.*}} : memref<?x?x?xf32, strided<[1, 0, ?]>>
  // CHECK: zuan.load %{{.*}} : memref<?x?x?xf32, strided<[0, 1, ?]>>
  // CHECK: zuan.store %{{.*}}, %{{.*}} : <?x?xf32>, memref<?x?xf32, strided<[?, 1]>>
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
  // CHECK: zuan.load %{{.*}} : memref<?x?x?xf32, strided<[?, 0, 1]>>
  // CHECK: zuan.load %{{.*}} : memref<?x?x?xf32, strided<[0, ?, 1]>>
  // CHECK: zuan.store %{{.*}}, %{{.*}} : <?x?xf32>, memref<?x?xf32, strided<[?, 1]>>
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                      ]
                      ins(%a, %b : memref<?x?xf32>, memref<?x?xf32>)
                      outs(%c: memref<?x?xf32>)
  return
}

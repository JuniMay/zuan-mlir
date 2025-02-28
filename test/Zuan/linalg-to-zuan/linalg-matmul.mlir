// RUN: zuan-opt -convert-linalg-to-zuan %s | FileCheck %s

// CHECK-LABEL: func.func @linalg_matmul
func.func @linalg_matmul(%a: memref<?x?xf32>, %b: memref<?x?xf32>, %c: memref<?x?xf32>) {
  // CHECK:      %[[A:.*]] = zuan.load %{{.*}} : memref<?x?x?xf32, strided<[?, 0, 1]>>
  // CHECK-NEXT: %[[B:.*]] = zuan.load %{{.*}} : memref<?x?x?xf32, strided<[0, 1, ?]>>
  // CHECK-NEXT: %[[MUL:.*]] = arith.mulf %[[A]], %[[B]] : !zuan.tile<?x?x?xf32>
  // CHECK-NEXT: %[[RED:.*]] = zuan.reduction <add> %[[MUL]] [2], %{{.*}} : !zuan.tile<?x?x?xf32>, !zuan.tile<?x?xf32>
  // CHECK-NEXT: zuan.yield {
  // CHECK-NEXT:   zuan.store %[[RED]], %{{.*}} : <?x?xf32>, memref<?x?xf32, strided<[?, 1]>
  // CHECK-NEXT: }
  linalg.matmul ins(%a, %b : memref<?x?xf32>, memref<?x?xf32>) outs(%c: memref<?x?xf32>)
  return 
}

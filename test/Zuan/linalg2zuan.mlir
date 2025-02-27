// RUN: zuan-opt -convert-linalg-to-zuan %s | FileCheck %s

// CHECK-LABEL: func.func @linalg_fill
func.func @linalg_fill(%a: f32, %buf: memref<?x?x?xf32>) {
  // CHECK:      zuan.dynamic(%[[TRANS:.*]] : memref<?x?x?xf32, strided<[?, ?, 1]>>) {
  // CHECK:        %[[SPLAT:.*]] = zuan.splat %{{.*}} [%{{.*}}, %{{.*}}, %{{.*}}] : f32
  // CHECK:        zuan.yield {
  // CHECK-NEXT:     zuan.store %[[SPLAT]], %[[TRANS]]
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  linalg.fill ins(%a: f32) outs(%buf: memref<?x?x?xf32>)
  return
}

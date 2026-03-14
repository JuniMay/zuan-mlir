// RUN: zuan-opt -convert-linalg-to-zuan %s | FileCheck %s

// CHECK-LABEL: func.func @linalg_fill
func.func @linalg_fill(%a: f32, %buf: memref<?x?x?xf32>) {
  // CHECK:      %[[TRANS:.*]] = memref.transpose
  // CHECK:      %[[SPLAT:.*]] = zuan.splat %{{.*}} [%{{.*}}, %{{.*}}, %{{.*}}] : f32
  // CHECK:      zuan.store %[[SPLAT]], %[[TRANS]]
  linalg.fill ins(%a: f32) outs(%buf: memref<?x?x?xf32>)
  return
}

// RUN: dyno-opt -convert-linalg-to-dyno %s | FileCheck %s

// CHECK-LABEL: func.func @linalg_fill
func.func @linalg_fill(%a: f32, %buf: memref<?x?x?xf32>) {
  // CHECK:      %[[TRANS:.*]] = memref.transpose
  // CHECK:      %[[SPLAT:.*]] = dyno.splat %{{.*}} [%{{.*}}, %{{.*}}, %{{.*}}] : f32
  // CHECK:      dyno.store %[[SPLAT]], %[[TRANS]]
  linalg.fill ins(%a: f32) outs(%buf: memref<?x?x?xf32>)
  return
}

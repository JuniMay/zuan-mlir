// RUN: zuan-opt -lower-zuan %s | FileCheck %s

// CHECK-LABEL: func.func @reduction
func.func @reduction(%a: memref<?x?x?x?xf32>, %b: memref<?x?xf32>) {
  zuan.dynamic (%b : memref<?x?xf32>) {
  ^bb0(%b_tile: !zuan.tile<?x?xf32>):
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: arith.addf
    // CHECK: scf.yield
    // CHECK: zuan.yield
    %a_tile = zuan.load %a : memref<?x?x?x?xf32>
    %reduced = zuan.reduction <add> %a_tile [1, 2], %b_tile : !zuan.tile<?x?x?x?xf32>, !zuan.tile<?x?xf32>
    zuan.yield {
      zuan.store %reduced, %b : !zuan.tile<?x?xf32>, memref<?x?xf32>
    }
  }
  return
}

// CHECK-LABEL: func.func @reduction_masked
func.func @reduction_masked(%a: memref<?x?x?x?xf32>, %b: memref<?x?xf32>, %m: memref<?x?x?x?xi1>) {
  zuan.dynamic (%b : memref<?x?xf32>) {
  ^bb0(%b_tile: !zuan.tile<?x?xf32>):
    %mask = zuan.load %m : memref<?x?x?x?xi1>
    %a_tile = zuan.load %a : memref<?x?x?x?xf32>
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: zuan.mask
    // CHECK: arith.addf
    // CHECK: zuan.mask_yield
    // CHECK: scf.yield
    // CHECK: scf.yield
    %reduced = zuan.mask %mask : !zuan.tile<?x?x?x?xi1> {
      %reduced = zuan.reduction <add> %a_tile [1, 2], %b_tile : !zuan.tile<?x?x?x?xf32>, !zuan.tile<?x?xf32>
      zuan.mask_yield %reduced : !zuan.tile<?x?xf32>
    } : !zuan.tile<?x?xf32>
    zuan.yield {
      zuan.store %reduced, %b : !zuan.tile<?x?xf32>, memref<?x?xf32>
    }
  }
  return
}

// CHECK-LABEL: func.func @reduction1d
func.func @reduction1d(%seq: memref<?xf32>) -> f32 {
  %res = zuan.dynamic (%seq : memref<?xf32>) {
  ^bb0(%seq_tile: !zuan.tile<?xf32>):
    // CHECK: zuan.reduction <add> %{{.*}} [0]
    %reduced = zuan.reduction <add> %seq_tile [0] : !zuan.tile<?xf32>
    zuan.yield %reduced : !zuan.tile<f32> {}
  } : f32
  return %res : f32
}

// CHECK-LABEL: func.func @reduction0d
func.func @reduction0d(%a: f32) -> f32 {
  %res = zuan.dynamic {
    %a_tile = zuan.splat %a [] : f32
    // CHECK: zuan.reduction <add> %{{.*}} []
    %reduced = zuan.reduction <add> %a_tile [] : !zuan.tile<f32>
    zuan.yield %reduced : !zuan.tile<f32> {}
  } : f32
  return %res : f32
}

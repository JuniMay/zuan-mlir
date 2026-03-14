// RUN: zuan-opt -lower-zuan -zuan-stripmining="vf=8 scalable=true" %s | FileCheck %s

// CHECK-LABEL: func.func @reduction
func.func @reduction(%a: memref<?x?x?x?xf32>, %b: memref<?x?xf32>) {
  %a_tile = zuan.load %a : memref<?x?x?x?xf32>
  %b_tile = zuan.load %b : memref<?x?xf32>
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: arith.addf
  %reduced = zuan.reduction <add> %a_tile [1, 2], %b_tile :
      !zuan.tile<?x?x?x?xf32>, !zuan.tile<?x?xf32>
  zuan.store %reduced, %b : !zuan.tile<?x?xf32>, memref<?x?xf32>
  return
}

// CHECK-LABEL: func.func @reduction_masked
func.func @reduction_masked(%a: memref<?x?x?x?xf32>, %b: memref<?x?xf32>,
                            %m: memref<?x?x?x?xi1>) {
  %mask = zuan.load %m : memref<?x?x?x?xi1>
  %a_tile = zuan.load %a : memref<?x?x?x?xf32>
  %b_tile = zuan.load %b : memref<?x?xf32>
  // CHECK: scf.for
  // CHECK: zuan.mask %{{.*}} {
  // CHECK: zuan.mask_yield %{{.*}}
  %reduced = zuan.mask %mask : !zuan.tile<?x?x?x?xi1> {
    %inner = zuan.reduction <add> %a_tile [1, 2], %b_tile :
        !zuan.tile<?x?x?x?xf32>, !zuan.tile<?x?xf32>
    zuan.mask_yield %inner : !zuan.tile<?x?xf32>
  } : !zuan.tile<?x?xf32>
  zuan.store %reduced, %b : !zuan.tile<?x?xf32>, memref<?x?xf32>
  return
}

// CHECK-LABEL: func.func @reduction1d
func.func @reduction1d(%seq: memref<?xf32>) -> f32 {
  %seq_tile = zuan.load %seq : memref<?xf32>
  // CHECK: scf.while
  %reduced = zuan.reduction <add> %seq_tile [0] : !zuan.tile<?xf32>
  // CHECK: zuan.extract
  %scalar = zuan.extract %reduced : !zuan.tile<f32>
  return %scalar : f32
}

// CHECK-LABEL: func.func @reduction0d
func.func @reduction0d(%a: f32) -> f32 {
  %a_tile = zuan.splat %a [] : f32
  // CHECK: zuan.reduction <add> %{{.*}} []
  %reduced = zuan.reduction <add> %a_tile [] : !zuan.tile<f32>
  %scalar = zuan.extract %reduced : !zuan.tile<f32>
  return %scalar : f32
}

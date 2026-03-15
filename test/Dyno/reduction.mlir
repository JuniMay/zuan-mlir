// RUN: dyno-opt -lower-dyno -dyno-stripmining="vf=8 scalable=true" %s | FileCheck %s

// CHECK-LABEL: func.func @reduction
func.func @reduction(%a: memref<?x?x?x?xf32>, %b: memref<?x?xf32>) {
  %a_tile = dyno.load %a : memref<?x?x?x?xf32>
  %b_tile = dyno.load %b : memref<?x?xf32>
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: scf.while
  // CHECK: dyno.reduction <add> %{{.*}} [0, 1], %{{.*}} : !dyno.tile<?x?x?xf32>, !dyno.tile<?xf32>
  %reduced = dyno.reduction <add> %a_tile [1, 2], %b_tile :
      !dyno.tile<?x?x?x?xf32>, !dyno.tile<?x?xf32>
  dyno.store %reduced, %b : !dyno.tile<?x?xf32>, memref<?x?xf32>
  return
}

// CHECK-LABEL: func.func @reduction_masked
func.func @reduction_masked(%a: memref<?x?x?x?xf32>, %b: memref<?x?xf32>,
                            %m: memref<?x?xi1>) {
  %mask = dyno.load %m : memref<?x?xi1>
  %a_tile = dyno.load %a : memref<?x?x?x?xf32>
  %b_tile = dyno.load %b : memref<?x?xf32>
  %inner = dyno.reduction <add> %a_tile [1, 2], %b_tile :
      !dyno.tile<?x?x?x?xf32>, !dyno.tile<?x?xf32>
  // CHECK: scf.for
  // CHECK: dyno.mask %{{.*}} {
  // CHECK: dyno.mask_yield %{{.*}}
  %reduced = dyno.mask %mask : !dyno.tile<?x?xi1> {
    dyno.mask_yield %inner : !dyno.tile<?x?xf32>
  } : !dyno.tile<?x?xf32>
  dyno.store %reduced, %b : !dyno.tile<?x?xf32>, memref<?x?xf32>
  return
}

// CHECK-LABEL: func.func @reduction1d
func.func @reduction1d(%seq: memref<?xf32>) -> f32 {
  %seq_tile = dyno.load %seq : memref<?xf32>
  // CHECK: scf.while
  %reduced = dyno.reduction <add> %seq_tile [0] : !dyno.tile<?xf32>
  // CHECK: dyno.extract
  %scalar = dyno.extract %reduced : !dyno.tile<f32>
  return %scalar : f32
}

// CHECK-LABEL: func.func @reduction0d
func.func @reduction0d(%a: f32) -> f32 {
  %a_tile = dyno.splat %a [] : f32
  // CHECK: dyno.reduction <add> %{{.*}} []
  %reduced = dyno.reduction <add> %a_tile [] : !dyno.tile<f32>
  %scalar = dyno.extract %reduced : !dyno.tile<f32>
  return %scalar : f32
}

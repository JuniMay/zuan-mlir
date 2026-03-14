// RUN: dyno-opt -lower-dyno -dyno-stripmining="vf=8 scalable=true" %s | FileCheck %s

// CHECK-LABEL: func.func @matmul
func.func @matmul(%a: memref<?x?xf32>, %b: memref<?x?xf32>,
                  %c: memref<?x?xf32>) {
  %a_tile = dyno.load %a : memref<?x?xf32>
  %b_tile = dyno.load %b : memref<?x?xf32>
  %c_tile = dyno.load %c : memref<?x?xf32>
  // CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[ACC:.+]] = %{{.+}})
  // CHECK: %[[A:.+]] = dyno.load
  // CHECK: %[[B:.+]] = dyno.load
  // CHECK: %[[OUTER:.+]] = dyno.outer <mul> %[[A]], %[[B]]
  // CHECK: %[[NEWACC:.+]] = arith.addf %[[ACC]], %[[OUTER]]
  %mm = dyno.matmul %a_tile, %b_tile : !dyno.tile<?x?xf32>, !dyno.tile<?x?xf32>
  %sum = arith.addf %mm, %c_tile : !dyno.tile<?x?xf32>
  dyno.store %sum, %c : !dyno.tile<?x?xf32>, memref<?x?xf32>
  return
}

// CHECK-LABEL: func.func @matmul2
func.func @matmul2(%a: memref<?x?xf32>, %b: memref<?x?xf32>,
                   %c: memref<?x?xf32>, %d: memref<?x?xf32>) {
  %a_tile = dyno.load %a : memref<?x?xf32>
  %b_tile = dyno.load %b : memref<?x?xf32>
  %c_tile = dyno.load %c : memref<?x?xf32>
  %d_tile = dyno.load %d : memref<?x?xf32>
  // CHECK: scf.for
  // CHECK: scf.for
  %mm1 = dyno.matmul %a_tile, %b_tile : !dyno.tile<?x?xf32>, !dyno.tile<?x?xf32>
  %mm2 = dyno.matmul %mm1, %c_tile : !dyno.tile<?x?xf32>, !dyno.tile<?x?xf32>
  %sum = arith.addf %mm2, %d_tile : !dyno.tile<?x?xf32>
  dyno.store %sum, %d : !dyno.tile<?x?xf32>, memref<?x?xf32>
  return
}

// CHECK-LABEL: func.func @matmul_masked
func.func @matmul_masked(%a: memref<?x?xf32>, %b: memref<?x?xf32>,
                         %c: memref<?x?xf32>, %m: memref<?x?xi1>) {
  %mask = dyno.load %m : memref<?x?xi1>
  %a_tile = dyno.mask %mask : !dyno.tile<?x?xi1> {
    %loaded = dyno.load %a : memref<?x?xf32>
    dyno.mask_yield %loaded : !dyno.tile<?x?xf32>
  } : !dyno.tile<?x?xf32>
  %b_tile = dyno.mask %mask : !dyno.tile<?x?xi1> {
    %loaded = dyno.load %b : memref<?x?xf32>
    dyno.mask_yield %loaded : !dyno.tile<?x?xf32>
  } : !dyno.tile<?x?xf32>
  %c_tile = dyno.load %c : memref<?x?xf32>
  // CHECK: scf.for
  // CHECK: dyno.mask %{{.*}} {
  // CHECK: dyno.mask_yield %{{.*}}
  %mm = dyno.mask %mask : !dyno.tile<?x?xi1> {
    %matmul = dyno.matmul %a_tile, %b_tile :
        !dyno.tile<?x?xf32>, !dyno.tile<?x?xf32>
    dyno.mask_yield %matmul : !dyno.tile<?x?xf32>
  } : !dyno.tile<?x?xf32>
  %sum = dyno.mask %mask : !dyno.tile<?x?xi1> {
    %add = arith.addf %mm, %c_tile : !dyno.tile<?x?xf32>
    dyno.mask_yield %add : !dyno.tile<?x?xf32>
  } : !dyno.tile<?x?xf32>
  dyno.mask %mask : !dyno.tile<?x?xi1> {
    dyno.store %sum, %c : !dyno.tile<?x?xf32>, memref<?x?xf32>
  }
  return
}

// CHECK-LABEL: func.func @matmul2_masked
func.func @matmul2_masked(%a: memref<?x?xf32>, %b: memref<?x?xf32>,
                          %c: memref<?x?xf32>, %m: memref<?x?xi1>,
                          %d: memref<?x?xf32>) {
  %mask = dyno.load %m : memref<?x?xi1>
  %a_tile = dyno.mask %mask : !dyno.tile<?x?xi1> {
    %loaded = dyno.load %a : memref<?x?xf32>
    dyno.mask_yield %loaded : !dyno.tile<?x?xf32>
  } : !dyno.tile<?x?xf32>
  %b_tile = dyno.mask %mask : !dyno.tile<?x?xi1> {
    %loaded = dyno.load %b : memref<?x?xf32>
    dyno.mask_yield %loaded : !dyno.tile<?x?xf32>
  } : !dyno.tile<?x?xf32>
  %c_tile = dyno.mask %mask : !dyno.tile<?x?xi1> {
    %loaded = dyno.load %c : memref<?x?xf32>
    dyno.mask_yield %loaded : !dyno.tile<?x?xf32>
  } : !dyno.tile<?x?xf32>
  %d_tile = dyno.load %d : memref<?x?xf32>
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: dyno.mask %{{.*}} {
  // CHECK: dyno.mask_yield %{{.*}}
  %mm1 = dyno.mask %mask : !dyno.tile<?x?xi1> {
    %matmul = dyno.matmul %a_tile, %b_tile :
        !dyno.tile<?x?xf32>, !dyno.tile<?x?xf32>
    dyno.mask_yield %matmul : !dyno.tile<?x?xf32>
  } : !dyno.tile<?x?xf32>
  %mm2 = dyno.mask %mask : !dyno.tile<?x?xi1> {
    %matmul = dyno.matmul %mm1, %c_tile :
        !dyno.tile<?x?xf32>, !dyno.tile<?x?xf32>
    dyno.mask_yield %matmul : !dyno.tile<?x?xf32>
  } : !dyno.tile<?x?xf32>
  %sum = dyno.mask %mask : !dyno.tile<?x?xi1> {
    %add = arith.addf %mm2, %d_tile : !dyno.tile<?x?xf32>
    dyno.mask_yield %add : !dyno.tile<?x?xf32>
  } : !dyno.tile<?x?xf32>
  dyno.mask %mask : !dyno.tile<?x?xi1> {
    dyno.store %sum, %d : !dyno.tile<?x?xf32>, memref<?x?xf32>
  }
  return
}

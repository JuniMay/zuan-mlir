// RUN: dyno-opt %s | FileCheck %s

// CHECK-LABEL: func.func @matmul_static_leading
func.func @matmul_static_leading(%a: memref<2x4x3xf32>, %b: memref<2x3x7xf32>,
                                 %c: memref<2x4x7xf32>) {
  %a_tile = dyno.load %a : memref<2x4x3xf32>
  %b_tile = dyno.load %b : memref<2x3x7xf32>
  %c_tile = dyno.load %c : memref<2x4x7xf32>
  %mm = dyno.matmul %a_tile, %b_tile :
      !dyno.tile<2x4x3xf32>, !dyno.tile<2x3x7xf32>
  // CHECK: arith.addf %{{.*}}, %{{.*}} : !dyno.tile<2x4x7xf32>
  %sum = arith.addf %mm, %c_tile : !dyno.tile<2x4x7xf32>
  dyno.store %sum, %c : !dyno.tile<2x4x7xf32>, memref<2x4x7xf32>
  return
}

// CHECK-LABEL: func.func @extract_type
func.func @extract_type(%arg0: f32) -> f32 {
  %tile = dyno.splat %arg0 [] : f32
  // CHECK: %[[SCALAR:.*]] = dyno.extract %{{.*}} : !dyno.tile<f32>
  %scalar = dyno.extract %tile : !dyno.tile<f32>
  return %scalar : f32
}

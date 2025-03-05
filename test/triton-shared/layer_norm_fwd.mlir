#map = affine_map<(d0) -> (d0)>
module {
  func.func @_layer_norm_fwd_fused(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: memref<*xf32> {tt.divisibility = 16 : i32}, %arg5: memref<*xf32> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: f32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c4096_i32 = arith.constant 4096 : i32
    %c4096 = arith.constant 4096 : index
    %cst_1 = arith.constant 0.000000e+00 : f16
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4096xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<4096xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_2 : memref<4096xf32>)
    %0 = arith.muli %arg12, %arg6 : i32
    %1 = arith.index_cast %0 : i32 to index
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<4096xf32>
    memref.copy %alloc_2, %alloc_3 : memref<4096xf32> to memref<4096xf32>
    %2 = scf.for %arg15 = %c0_i32 to %arg7 step %c4096_i32 iter_args(%arg16 = %alloc_3) -> (memref<4096xf32>)  : i32 {
      %13 = arith.index_cast %arg15 : i32 to index
      %14 = arith.addi %1, %13 : index
      %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%14], sizes: [4096], strides: [1] : memref<*xf16> to memref<4096xf16, strided<[1], offset: ?>>
      %15 = arith.addi %13, %c4096 : index
      %16 = arith.index_cast %arg7 : i32 to index
      %17 = arith.minsi %15, %16 : index
      %18 = arith.maxsi %17, %13 : index
      %19 = arith.subi %18, %13 : index
      %alloc_8 = memref.alloc() : memref<4096xf16>
      %20 = arith.cmpi slt, %19, %c4096 : index
      scf.if %20 {
        linalg.fill ins(%cst_1 : f16) outs(%alloc_8 : memref<4096xf16>)
      }
      %subview = memref.subview %reinterpret_cast_7[0] [%19] [1] : memref<4096xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_9 = memref.subview %alloc_8[0] [%19] [1] : memref<4096xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview, %subview_9 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg16, %alloc_8 : memref<4096xf32>, memref<4096xf16>) outs(%alloc : memref<4096xf32>) {
      ^bb0(%in: f32, %in_11: f16, %out: f32):
        %21 = arith.extf %in_11 : f16 to f32
        %22 = arith.addf %in, %21 : f32
        linalg.yield %22 : f32
      }
      %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<4096xf32>
      memref.copy %alloc, %alloc_10 : memref<4096xf32> to memref<4096xf32>
      scf.yield %alloc_10 : memref<4096xf32>
    }
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_4[] : memref<f32>
    linalg.reduce ins(%2 : memref<4096xf32>) outs(%alloc_4 : memref<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %13 = arith.addf %in, %init : f32
        linalg.yield %13 : f32
      }
    %3 = memref.load %alloc_4[] : memref<f32>
    %4 = arith.sitofp %arg7 : i32 to f32
    %5 = arith.divf %3, %4 : f32
    %6 = scf.for %arg15 = %c0_i32 to %arg7 step %c4096_i32 iter_args(%arg16 = %alloc_2) -> (memref<4096xf32>)  : i32 {
      %13 = arith.index_cast %arg15 : i32 to index
      %14 = arith.addi %1, %13 : index
      %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%14], sizes: [4096], strides: [1] : memref<*xf16> to memref<4096xf16, strided<[1], offset: ?>>
      %15 = arith.addi %13, %c4096 : index
      %16 = arith.index_cast %arg7 : i32 to index
      %17 = arith.minsi %15, %16 : index
      %18 = arith.maxsi %17, %13 : index
      %19 = arith.subi %18, %13 : index
      %alloc_8 = memref.alloc() : memref<4096xf16>
      %20 = arith.cmpi slt, %19, %c4096 : index
      scf.if %20 {
        linalg.fill ins(%cst_1 : f16) outs(%alloc_8 : memref<4096xf16>)
      }
      %subview = memref.subview %reinterpret_cast_7[0] [%19] [1] : memref<4096xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_9 = memref.subview %alloc_8[0] [%19] [1] : memref<4096xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview, %subview_9 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg16, %alloc_8 : memref<4096xf32>, memref<4096xf16>) outs(%alloc : memref<4096xf32>) {
      ^bb0(%in: f32, %in_11: f16, %out: f32):
        %21 = linalg.index 0 : index
        %22 = arith.extf %in_11 : f16 to f32
        %23 = arith.subf %22, %5 : f32
        %24 = arith.index_cast %21 : index to i32
        %25 = arith.addi %arg15, %24 : i32
        %26 = arith.cmpi slt, %25, %arg7 : i32
        %27 = arith.select %26, %23, %cst : f32
        %28 = arith.mulf %27, %27 : f32
        %29 = arith.addf %in, %28 : f32
        linalg.yield %29 : f32
      }
      %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<4096xf32>
      memref.copy %alloc, %alloc_10 : memref<4096xf32> to memref<4096xf32>
      scf.yield %alloc_10 : memref<4096xf32>
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_5[] : memref<f32>
    linalg.reduce ins(%6 : memref<4096xf32>) outs(%alloc_5 : memref<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %13 = arith.addf %in, %init : f32
        linalg.yield %13 : f32
      }
    %7 = memref.load %alloc_5[] : memref<f32>
    %8 = arith.divf %7, %4 : f32
    %9 = arith.addf %8, %arg8 : f32
    %10 = math.sqrt %9 : f32
    %11 = arith.divf %cst_0, %10 : f32
    %12 = arith.index_cast %arg12 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%12], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    affine.store %5, %reinterpret_cast[0] : memref<1xf32, strided<[1], offset: ?>>
    %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%12], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    affine.store %11, %reinterpret_cast_6[0] : memref<1xf32, strided<[1], offset: ?>>
    scf.for %arg15 = %c0_i32 to %arg7 step %c4096_i32  : i32 {
      %13 = arith.index_cast %arg15 : i32 to index
      %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%13], sizes: [4096], strides: [1] : memref<*xf16> to memref<4096xf16, strided<[1], offset: ?>>
      %14 = arith.addi %13, %c4096 : index
      %15 = arith.index_cast %arg7 : i32 to index
      %16 = arith.minsi %14, %15 : index
      %17 = arith.maxsi %16, %13 : index
      %18 = arith.subi %17, %13 : index
      %alloc_8 = memref.alloc() : memref<4096xf16>
      %subview = memref.subview %reinterpret_cast_7[0] [%18] [1] : memref<4096xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_9 = memref.subview %alloc_8[0] [%18] [1] : memref<4096xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview, %subview_9 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%13], sizes: [4096], strides: [1] : memref<*xf16> to memref<4096xf16, strided<[1], offset: ?>>
      %alloc_11 = memref.alloc() : memref<4096xf16>
      %subview_12 = memref.subview %reinterpret_cast_10[0] [%18] [1] : memref<4096xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_13 = memref.subview %alloc_11[0] [%18] [1] : memref<4096xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview_12, %subview_13 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      %19 = arith.addi %1, %13 : index
      %reinterpret_cast_14 = memref.reinterpret_cast %arg0 to offset: [%19], sizes: [4096], strides: [1] : memref<*xf16> to memref<4096xf16, strided<[1], offset: ?>>
      %alloc_15 = memref.alloc() : memref<4096xf16>
      %20 = arith.cmpi slt, %18, %c4096 : index
      scf.if %20 {
        linalg.fill ins(%cst_1 : f16) outs(%alloc_15 : memref<4096xf16>)
      }
      %subview_16 = memref.subview %reinterpret_cast_14[0] [%18] [1] : memref<4096xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_17 = memref.subview %alloc_15[0] [%18] [1] : memref<4096xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview_16, %subview_17 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      %reinterpret_cast_18 = memref.reinterpret_cast %arg1 to offset: [%19], sizes: [4096], strides: [1] : memref<*xf16> to memref<4096xf16, strided<[1], offset: ?>>
      %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<4096xf16>
      linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%alloc_15, %alloc_8, %alloc_11 : memref<4096xf16>, memref<4096xf16>, memref<4096xf16>) outs(%alloc_19 : memref<4096xf16>) {
      ^bb0(%in: f16, %in_22: f16, %in_23: f16, %out: f16):
        %21 = arith.extf %in : f16 to f32
        %22 = arith.subf %21, %5 : f32
        %23 = arith.mulf %22, %11 : f32
        %24 = arith.extf %in_22 : f16 to f32
        %25 = arith.mulf %23, %24 : f32
        %26 = arith.extf %in_23 : f16 to f32
        %27 = arith.addf %25, %26 : f32
        %28 = arith.truncf %27 : f32 to f16
        linalg.yield %28 : f16
      }
      %subview_20 = memref.subview %alloc_19[0] [%18] [1] : memref<4096xf16> to memref<?xf16, strided<[1]>>
      %subview_21 = memref.subview %reinterpret_cast_18[0] [%18] [1] : memref<4096xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      memref.copy %subview_20, %subview_21 : memref<?xf16, strided<[1]>> to memref<?xf16, strided<[1], offset: ?>>
    }
    return
  }
}


#map = affine_map<(d0) -> (d0)>
module {
  func.func @complex_gather_scatter(%arg0: memref<*xi32> {tt.divisibility = 16 : i32}, %arg1: memref<*xi32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c32_i32 = arith.constant 32 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xi32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32xi32>
    linalg.fill ins(%c32_i32 : i32) outs(%alloc_0 : memref<32xi32>)
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<32xi32>
    linalg.fill ins(%c3_i32 : i32) outs(%alloc_1 : memref<32xi32>)
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<32xi32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloc_2 : memref<32xi32>) {
    ^bb0(%out: i32):
      %1 = linalg.index 0 : index
      %2 = arith.index_cast %1 : index to i32
      linalg.yield %2 : i32
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32xi32>
    memref.copy %alloc_2, %alloc_3 : memref<32xi32> to memref<32xi32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32xi32>
    memref.copy %alloc_2, %alloc_4 : memref<32xi32> to memref<32xi32>
    %0:3 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %alloc_3, %arg10 = %alloc_4, %arg11 = %c0) -> (memref<32xi32>, memref<32xi32>, index)  : i32 {
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg9, %alloc_1 : memref<32xi32>, memref<32xi32>) outs(%arg9 : memref<32xi32>) {
      ^bb0(%in: i32, %in_8: i32, %out: i32):
        %5 = arith.divsi %in, %in_8 : i32
        linalg.yield %5 : i32
      }
      %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32xi32>
      linalg.fill ins(%arg8 : i32) outs(%alloc_5 : memref<32xi32>)
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg9, %alloc_5 : memref<32xi32>, memref<32xi32>) outs(%arg9 : memref<32xi32>) {
      ^bb0(%in: i32, %in_8: i32, %out: i32):
        %5 = arith.addi %in, %in_8 : i32
        linalg.yield %5 : i32
      }
      %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<32xi1>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg9, %alloc_0 : memref<32xi32>, memref<32xi32>) outs(%alloc_6 : memref<32xi1>) {
      ^bb0(%in: i32, %in_8: i32, %out: i1):
        %5 = arith.cmpi slt, %in, %in_8 : i32
        linalg.yield %5 : i1
      }
      %cast = memref.cast %arg0 : memref<*xi32> to memref<?xi32>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg9, %alloc_6 : memref<32xi32>, memref<32xi1>) outs(%alloc : memref<32xi32>) {
      ^bb0(%in: i32, %in_8: i1, %out: i32):
        %5 = scf.if %in_8 -> (i32) {
          %6 = arith.index_cast %in : i32 to index
          %7 = memref.load %cast[%6] : memref<?xi32>
          scf.yield %7 : i32
        } else {
          scf.yield %c0_i32 : i32
        }
        linalg.yield %5 : i32
      }
      %cast_7 = memref.cast %arg1 : memref<*xi32> to memref<?xi32>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg9, %alloc : memref<32xi32>, memref<32xi32>) {
      ^bb0(%in: i32, %in_8: i32):
        %5 = arith.index_cast %in : i32 to index
        memref.store %in_8, %cast_7[%5] : memref<?xi32>
        linalg.yield
      }
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg9, %alloc_0 : memref<32xi32>, memref<32xi32>) outs(%arg9 : memref<32xi32>) {
      ^bb0(%in: i32, %in_8: i32, %out: i32):
        %5 = arith.addi %in, %in_8 : i32
        linalg.yield %5 : i32
      }
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg10, %alloc_0 : memref<32xi32>, memref<32xi32>) outs(%arg10 : memref<32xi32>) {
      ^bb0(%in: i32, %in_8: i32, %out: i32):
        %5 = arith.addi %in, %in_8 : i32
        linalg.yield %5 : i32
      }
      %1 = arith.addi %arg8, %c1_i32 : i32
      %2 = arith.addi %arg11, %c32 : index
      %3:3 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %arg9, %arg14 = %arg10, %arg15 = %2) -> (memref<32xi32>, memref<32xi32>, index)  : i32 {
        %5 = arith.addi %arg12, %c1_i32 : i32
        %6 = arith.muli %1, %5 : i32
        linalg.fill ins(%6 : i32) outs(%alloc : memref<32xi32>)
        linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg13, %alloc : memref<32xi32>, memref<32xi32>) outs(%arg13 : memref<32xi32>) {
        ^bb0(%in: i32, %in_8: i32, %out: i32):
          %8 = arith.divsi %in, %in_8 : i32
          linalg.yield %8 : i32
        }
        linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg13, %alloc_5 : memref<32xi32>, memref<32xi32>) outs(%arg13 : memref<32xi32>) {
        ^bb0(%in: i32, %in_8: i32, %out: i32):
          %8 = arith.addi %in, %in_8 : i32
          linalg.yield %8 : i32
        }
        linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg13, %alloc_0 : memref<32xi32>, memref<32xi32>) outs(%alloc_6 : memref<32xi1>) {
        ^bb0(%in: i32, %in_8: i32, %out: i1):
          %8 = arith.cmpi slt, %in, %in_8 : i32
          linalg.yield %8 : i1
        }
        linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg13, %alloc_6 : memref<32xi32>, memref<32xi1>) outs(%alloc : memref<32xi32>) {
        ^bb0(%in: i32, %in_8: i1, %out: i32):
          %8 = scf.if %in_8 -> (i32) {
            %9 = arith.index_cast %in : i32 to index
            %10 = memref.load %cast[%9] : memref<?xi32>
            scf.yield %10 : i32
          } else {
            scf.yield %c0_i32 : i32
          }
          linalg.yield %8 : i32
        }
        linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg13, %alloc : memref<32xi32>, memref<32xi32>) {
        ^bb0(%in: i32, %in_8: i32):
          %8 = arith.index_cast %in : i32 to index
          memref.store %in_8, %cast_7[%8] : memref<?xi32>
          linalg.yield
        }
        linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg13, %alloc_0 : memref<32xi32>, memref<32xi32>) outs(%arg13 : memref<32xi32>) {
        ^bb0(%in: i32, %in_8: i32, %out: i32):
          %8 = arith.addi %in, %in_8 : i32
          linalg.yield %8 : i32
        }
        linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg14, %alloc_0 : memref<32xi32>, memref<32xi32>) outs(%arg14 : memref<32xi32>) {
        ^bb0(%in: i32, %in_8: i32, %out: i32):
          %8 = arith.addi %in, %in_8 : i32
          linalg.yield %8 : i32
        }
        %7 = arith.addi %arg15, %c32 : index
        scf.yield %arg13, %arg14, %7 : memref<32xi32>, memref<32xi32>, index
      }
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%3#0, %alloc_1 : memref<32xi32>, memref<32xi32>) outs(%3#0 : memref<32xi32>) {
      ^bb0(%in: i32, %in_8: i32, %out: i32):
        %5 = arith.divsi %in, %in_8 : i32
        linalg.yield %5 : i32
      }
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%3#0, %alloc_5 : memref<32xi32>, memref<32xi32>) outs(%3#0 : memref<32xi32>) {
      ^bb0(%in: i32, %in_8: i32, %out: i32):
        %5 = arith.addi %in, %in_8 : i32
        linalg.yield %5 : i32
      }
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%3#0, %alloc_0 : memref<32xi32>, memref<32xi32>) outs(%alloc_6 : memref<32xi1>) {
      ^bb0(%in: i32, %in_8: i32, %out: i1):
        %5 = arith.cmpi slt, %in, %in_8 : i32
        linalg.yield %5 : i1
      }
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%3#0, %alloc_6 : memref<32xi32>, memref<32xi1>) outs(%alloc : memref<32xi32>) {
      ^bb0(%in: i32, %in_8: i1, %out: i32):
        %5 = scf.if %in_8 -> (i32) {
          %6 = arith.index_cast %in : i32 to index
          %7 = memref.load %cast[%6] : memref<?xi32>
          scf.yield %7 : i32
        } else {
          scf.yield %c0_i32 : i32
        }
        linalg.yield %5 : i32
      }
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%3#0, %alloc : memref<32xi32>, memref<32xi32>) {
      ^bb0(%in: i32, %in_8: i32):
        %5 = arith.index_cast %in : i32 to index
        memref.store %in_8, %cast_7[%5] : memref<?xi32>
        linalg.yield
      }
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%3#0, %alloc_0 : memref<32xi32>, memref<32xi32>) outs(%3#0 : memref<32xi32>) {
      ^bb0(%in: i32, %in_8: i32, %out: i32):
        %5 = arith.addi %in, %in_8 : i32
        linalg.yield %5 : i32
      }
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%3#1, %alloc_0 : memref<32xi32>, memref<32xi32>) outs(%3#1 : memref<32xi32>) {
      ^bb0(%in: i32, %in_8: i32, %out: i32):
        %5 = arith.addi %in, %in_8 : i32
        linalg.yield %5 : i32
      }
      %4 = arith.addi %3#2, %c32 : index
      scf.yield %3#0, %3#1, %4 : memref<32xi32>, memref<32xi32>, index
    }
    return
  }
}


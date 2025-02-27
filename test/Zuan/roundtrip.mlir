// RUN: zuan-opt -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @empty
func.func @empty() {
  zuan.dynamic {
    zuan.yield {}
  }
  return
}

// CHECK-LABEL: func.func @roundtrip
func.func @roundtrip(%c: memref<?x?xf32>) {
  zuan.dynamic (%c : memref<?x?xf32>) {
  ^bb0(%arg: !zuan.tile<?x?xf32>):
    zuan.yield {
      zuan.store %arg, %c : !zuan.tile<?x?xf32>, memref<?x?xf32>
    }
  }
  return
}

// CHECK-LABEL: func.func @matmul
func.func @matmul(%a: memref<?x?xf32>, %b: memref<?x?xf32>, %c: memref<?x?xf32>) {
  zuan.dynamic (%c : memref<?x?xf32>) {
  ^bb0(%c_tile: !zuan.tile<?x?xf32>):
    %a_tile = zuan.load %a : memref<?x?xf32>
    %b_tile = zuan.load %b : memref<?x?xf32>
    %mm = zuan.matmul %a_tile, %b_tile : !zuan.tile<?x?xf32>, !zuan.tile<?x?xf32>
    %add = arith.addf %mm, %c_tile : !zuan.tile<?x?xf32>
    zuan.yield {
      zuan.store %add, %c : !zuan.tile<?x?xf32>, memref<?x?xf32>
    }
  }
  return
}

// CHECK-LABEL: func.func @multi_reduction
func.func @multi_reduction(%a: memref<?x?x?x?xf32>, %b: memref<?x?xf32>) {
  zuan.dynamic (%b : memref<?x?xf32>) {
  ^bb0(%b_tile: !zuan.tile<?x?xf32>):
    %a_tile = zuan.load %a : memref<?x?x?x?xf32>
    %reduced = zuan.multi_reduction <add> %a_tile [1, 2], %b_tile : !zuan.tile<?x?x?x?xf32>, !zuan.tile<?x?xf32>
    zuan.yield {
      zuan.store %reduced, %b : !zuan.tile<?x?xf32>, memref<?x?xf32>
    }
  }
  return
}

// CHECK-LABEL: func.func @splat
func.func @splat(%a: memref<?x?xf32>, %b: memref<?x4xf32>, %c: memref<4x?x?xf32>) -> f32 {
  %cst = arith.constant 1.0 : f32
  %0 = arith.constant 0 : index
  %dim = memref.dim %b, %0 : memref<?x4xf32>
  %res = zuan.dynamic (%b, %c : memref<?x4xf32>, memref<4x?x?xf32>) {
  ^bb0(%b_tile: !zuan.tile<?x4xf32>, %c_tile: !zuan.tile<4x?x?xf32>):
    // splat scalar to tile
    %splat_cst = zuan.splat %cst [%dim, 4] : f32
    // splat tile to tile (broadcast)
    %a_tile = zuan.load %a : memref<?x?xf32>
    %a_splat = zuan.splat %a_tile [4] : !zuan.tile<?x?xf32>
    // writeback to memref, yield the scalar for testing
    zuan.yield %cst : f32 {
      zuan.store %splat_cst, %b : !zuan.tile<?x4xf32>, memref<?x4xf32>
      zuan.store %a_splat, %c : !zuan.tile<4x?x?xf32>, memref<4x?x?xf32>
    }
  } : f32
  return %res : f32
}

// CHECK-LABEL: func.func @outer_samerank
func.func @outer_samerank(%a: memref<?x4xf32>, %b: memref<?x7xf32>, %c: memref<?x?x?xf32>) {
  zuan.dynamic (%c : memref<?x?x?xf32>) {
  ^bb0(%c_tile: !zuan.tile<?x4x7xf32>):
    %a_tile = zuan.load %a : memref<?x4xf32>
    %b_tile = zuan.load %b : memref<?x7xf32>
    %outer = zuan.outer <add> %a_tile, %b_tile : !zuan.tile<?x4xf32>, !zuan.tile<?x7xf32>
    %add = arith.addf %outer, %c_tile : !zuan.tile<?x4x7xf32>
    zuan.yield {
      %c_cast = memref.cast %c : memref<?x?x?xf32> to memref<?x4x7xf32>
      zuan.store %add, %c_cast : !zuan.tile<?x4x7xf32>, memref<?x4x7xf32>
    }
  }
  return
}

// CHECK-LABEL: func.func @outer_diffrank
func.func @outer_diffrank(%a: memref<?x4xf32>, %b: memref<?xf32>, %c: memref<?x?xf32>) {
  zuan.dynamic (%c : memref<?x?xf32>) {
  ^bb0(%c_tile: !zuan.tile<?x4xf32>):
    %a_tile = zuan.load %a : memref<?x4xf32>
    %b_tile = zuan.load %b : memref<?xf32>
    %outer = zuan.outer <add> %a_tile, %b_tile : !zuan.tile<?x4xf32>, !zuan.tile<?xf32>
    %add = arith.addf %outer, %c_tile : !zuan.tile<?x4xf32>
    zuan.yield {
      %c_cast = memref.cast %c : memref<?x?xf32> to memref<?x4xf32>
      zuan.store %add, %c_cast : !zuan.tile<?x4xf32>, memref<?x4xf32>
    }
  }
  return
}

// CHECK-LABEL: func.func @masking
func.func @masking(%a: memref<?x?xf32>, %b: memref<?x?xf32>, %c: memref<?x?xf32>) {
  zuan.dynamic (%c : memref<?x?xf32>) {
  ^bb0(%c_tile: !zuan.tile<?x?xf32>):
    %a_tile = zuan.load %a : memref<?x?xf32>
    %b_tile = zuan.load %b : memref<?x?xf32>
    
    %slt = arith.cmpf olt, %a_tile, %b_tile : !zuan.tile<?x?xf32> 
    %masked_add = zuan.mask %slt : !zuan.tile<?x?xi1> {
      %add = arith.addf %a_tile, %b_tile : !zuan.tile<?x?xf32>
      zuan.mask_yield %add : !zuan.tile<?x?xf32>
    } : !zuan.tile<?x?xf32>

    zuan.yield {
      zuan.mask %slt : !zuan.tile<?x?xi1> {
        zuan.store %masked_add, %c : !zuan.tile<?x?xf32>, memref<?x?xf32>
      }
    }
  }
  return
}

// CHECK-LABEL: func.func @gather_scatter
func.func @gather_scatter(
  %a : memref<?x?xf32>, %a_idx0: memref<?x?x?xindex>, %a_idx1: memref<?x?x?xindex>, 
  %b : memref<?x?xf32>, %b_idx0: memref<?x?x?xindex>, %b_idx1: memref<?x?x?xindex>) {
  zuan.dynamic (%b : memref<?x?xf32>) {
  ^bb0(%b_tile: !zuan.tile<?x?xf32>):
    %a_idx0_tile = zuan.load %a_idx0 : memref<?x?x?xindex>
    %a_idx1_tile = zuan.load %a_idx1 : memref<?x?x?xindex>
    %b_idx0_tile = zuan.load %b_idx0 : memref<?x?x?xindex>
    %b_idx1_tile = zuan.load %b_idx1 : memref<?x?x?xindex>
    %gathered = zuan.gather %a[%a_idx0_tile, %a_idx1_tile] : memref<?x?xf32>, !zuan.tile<?x?x?xindex>, !zuan.tile<?x?x?xindex>
    zuan.yield {
      zuan.scatter %gathered, %b[%b_idx0_tile, %b_idx1_tile] : !zuan.tile<?x?x?xf32>, memref<?x?xf32>, !zuan.tile<?x?x?xindex>, !zuan.tile<?x?x?xindex>
    }
  }
  return
}

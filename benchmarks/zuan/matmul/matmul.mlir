func.func @matmul(%a: memref<?x?xf32>, %b: memref<?x?xf32>,
                  %c: memref<?x?xf32>) {
  %a_tile = zuan.load %a : memref<?x?xf32>
  %b_tile = zuan.load %b : memref<?x?xf32>
  %c_tile = zuan.load %c : memref<?x?xf32>
  %mm = zuan.matmul %a_tile, %b_tile : !zuan.tile<?x?xf32>, !zuan.tile<?x?xf32>
  %sum = arith.addf %mm, %c_tile : !zuan.tile<?x?xf32>
  zuan.store %sum, %c : !zuan.tile<?x?xf32>, memref<?x?xf32>
  return
}

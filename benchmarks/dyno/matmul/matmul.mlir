func.func @matmul(%a: memref<?x?xf32>, %b: memref<?x?xf32>,
                  %c: memref<?x?xf32>) {
  %a_tile = dyno.load %a : memref<?x?xf32>
  %b_tile = dyno.load %b : memref<?x?xf32>
  %c_tile = dyno.load %c : memref<?x?xf32>
  %mm = dyno.matmul %a_tile, %b_tile : !dyno.tile<?x?xf32>, !dyno.tile<?x?xf32>
  %sum = arith.addf %mm, %c_tile : !dyno.tile<?x?xf32>
  dyno.store %sum, %c : !dyno.tile<?x?xf32>, memref<?x?xf32>
  return
}

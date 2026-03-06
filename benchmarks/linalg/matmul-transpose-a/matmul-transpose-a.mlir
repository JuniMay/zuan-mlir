func.func @kernel(%a: memref<?x?xf32>, %b: memref<?x?xf32>, %c: memref<?x?xf32>) {
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d2, d0)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                      ]
                      ins(%a, %b : memref<?x?xf32>, memref<?x?xf32>)
                      outs(%c: memref<?x?xf32>)
  return 
}

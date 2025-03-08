func.func @kernel(%a: memref<?x?xf32>, %b: memref<?x?xf32>, %c: memref<?x?xf32>) {
  linalg.matmul_transpose_a ins(%a, %b : memref<?x?xf32>, memref<?x?xf32>) outs(%c: memref<?x?xf32>)
  return 
}

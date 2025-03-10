func.func @kernel(%a: memref<?x?xf16>, %b: memref<?x?xf16>, %c: memref<?x?xf16>) {
  linalg.matmul ins(%a, %b : memref<?x?xf16>, memref<?x?xf16>) outs(%c: memref<?x?xf16>)
  return 
}

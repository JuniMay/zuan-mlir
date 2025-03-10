
func.func @kernel(%a: memref<?x?x?x?xf32>, %dst: memref<?x?x?x?xf32>) {
  linalg.rsqrt ins(%a: memref<?x?x?x?xf32>) outs(%dst: memref<?x?x?x?xf32>)
  return
}

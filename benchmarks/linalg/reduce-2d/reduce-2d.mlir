func.func @kernel(%a: memref<?x?xf32>, %dst: memref<?xf32>) {
  linalg.reduce { arith.addf } ins(%a: memref<?x?xf32>) outs(%dst: memref<?xf32>) dimensions=[0]
  return
}

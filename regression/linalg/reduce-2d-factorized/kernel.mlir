func.func @kernel(%a: memref<?x?xi32>, %dst: memref<?xi32>) {
  linalg.reduce { arith.addi } ins(%a: memref<?x?xi32>) outs(%dst: memref<?xi32>) dimensions=[0]
  return
}

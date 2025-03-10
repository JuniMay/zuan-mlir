func.func @kernel(%a: memref<?xf32>, %dst: memref<f32>) {
  linalg.reduce { arith.addf } ins(%a: memref<?xf32>) outs(%dst: memref<f32>) dimensions=[0]
  return
}

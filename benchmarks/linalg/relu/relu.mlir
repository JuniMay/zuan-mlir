func.func @relu_kernel(%a: memref<?x?x?x?xf32>, %b: memref<?x?x?x?xf32>) {
  linalg.generic {
    indexing_maps = [
      affine_map<(i, j, k, l) -> (i, j, k, l)>,
      affine_map<(i, j, k, l) -> (i, j, k, l)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  }
  ins(%a : memref<?x?x?x?xf32>) outs(%b: memref<?x?x?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %cst0 = arith.constant 0.0 : f32
    %cond = arith.cmpf olt, %in, %cst0 : f32
    %res = arith.select %cond, %cst0, %in : f32
    linalg.yield %res : f32
  }
  return
}

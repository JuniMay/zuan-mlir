func.func @kernel(%min: f64, %max: f64, %seed: i32, %output: memref<?x?xf32>) {
  linalg.fill_rng_2d ins(%min, %max, %seed : f64, f64, i32) outs(%output: memref<?x?xf32>)
  return
}

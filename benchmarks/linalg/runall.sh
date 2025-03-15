#!/bin/bash

# get build-directory from command-line and data-directory
if [ $# -ne 2 ]; then
  echo "Usage: $0 <build-dir> <data-dir>"
  exit 1
fi

BUILD_DIR=$1
DATA_DIR=$2

BENCHMARKS=(
  "dot-fp16"
  "fill-rng-2d"
  "matmul"
  "matmul-fp16"
  "matmul-transpose-a"
  "matmul-transpose-b"
  "mmt4d"
  "quantized-matmul"
  "reduce"
  "reduce-2d"
  "relu"
  "rsqrt"
  "exp"
)

for BENCHMARK in "${BENCHMARKS[@]}"; do
  $BUILD_DIR/benchmarks/linalg/$BENCHMARK/linalg-$BENCHMARK-benchmark --benchmark_format=csv > $DATA_DIR/linalg-$BENCHMARK.csv
done

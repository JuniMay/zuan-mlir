#include "benchmark/benchmark.h"
#include "common/MemRefUtils.h"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <random>

extern "C" {
void _mlir_ciface_relu_kernel_autovec(MemRef<float, 4> *, MemRef<float, 4> *);
void _mlir_ciface_relu_kernel_zuan(MemRef<float, 4> *, MemRef<float, 4> *);
}

using KernelFunc = void (*)(MemRef<float, 4> *, MemRef<float, 4> *);

static auto initializeData(size_t b, size_t h, size_t w, size_t c) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);

  MemRef<float, 4> input({b, h, w, c}, 0);

  for (size_t i = 0; i < b; i++) {
    for (size_t j = 0; j < h; j++) {
      for (size_t k = 0; k < w; k++) {
        for (size_t l = 0; l < c; l++) {
          input[i * h * w * c + j * w * c + k * c + l] = dis(gen);
        }
      }
    }
  }

  return input;
}

static void runKernel(KernelFunc kernel, MemRef<float, 4> *input,
                      MemRef<float, 4> *output) {
  kernel(input, output);
}

static void runBenchmark(benchmark::State &state, KernelFunc kernel) {
  size_t b = state.range(0);
  size_t h = state.range(1);
  size_t w = state.range(2);
  size_t c = state.range(3);

  MemRef<float, 4> input = initializeData(b, h, w, c);
  MemRef<float, 4> output({b, h, w, c}, 0);
  for (auto _ : state) {
    state.PauseTiming();
    output.fill(0);
    state.ResumeTiming();
    runKernel(kernel, &input, &output);
  }
}

static void verifyRelu() {
  const size_t B = 64;
  const size_t H = 56;
  const size_t W = 56;
  const size_t C = 64;

  MemRef<float, 4> input = initializeData(B, H, W, C);
  MemRef<float, 4> output1({B, H, W, C}, 0);
  MemRef<float, 4> output2({B, H, W, C}, 0);

  runKernel(_mlir_ciface_relu_kernel_autovec, &input, &output1);
  runKernel(_mlir_ciface_relu_kernel_zuan, &input, &output2);

  output1.verify(output2, "Relu", 0);

  for (size_t i = 0; i < 10; i++) {
    std::cout << "Index " << i << ":\tAutovec = " << std::setprecision(10)
              << output1[i] << "\tZuan = " << std::setprecision(10)
              << output2[i] << std::endl;
  }
}

BENCHMARK_CAPTURE(runBenchmark, zuan, _mlir_ciface_relu_kernel_zuan)
    ->Unit(benchmark::kMillisecond)
    ->Args({64, 56, 56, 64})
    ->Args({64, 96, 96, 128})
    // Irregular shapes
    ->Args({1, 3, 171, 103})
    ->Args({179, 253, 19, 129});

BENCHMARK_CAPTURE(runBenchmark, autovec, _mlir_ciface_relu_kernel_autovec)
    ->Unit(benchmark::kMillisecond)
    ->Args({64, 56, 56, 64})
    ->Args({64, 96, 96, 128})
    // Irregular shapes
    ->Args({1, 3, 171, 103})
    ->Args({179, 253, 19, 129});

int main(int argc, char **argv) {
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  std::cout << "------------------------------------------------" << std::endl;
  verifyRelu();
  std::cout << "------------------------------------------------" << std::endl;
  return 0;
}
#include "benchmark/benchmark.h"
#include "common/MemRefUtils.h"
#include <cassert>
#include <iomanip>
#include <iostream>

extern "C" {
void _mlir_ciface_kernel_autovec(double, double, int, MemRef<float, 2> *);
void _mlir_ciface_kernel_zuan(double, double, int, MemRef<float, 2> *);
}

using KernelFunc = void (*)(double, double, int, MemRef<float, 2> *);

static void runKernel(KernelFunc kernel, double min, double max, int seed,
                      MemRef<float, 2> *output) {
  kernel(min, max, seed, output);
}

static void runBenchmark(benchmark::State &state, KernelFunc kernel) {
  size_t m = state.range(0);
  size_t n = state.range(1);

  MemRef<float, 2> output({m, n}, 0);
  double min = -1.0;
  double max = 1.0;
  int seed = 19260817;
  for (auto _ : state) {
    runKernel(kernel, min, max, seed, &output);
  }
}

static void verifyFillRng2D() {
  const size_t M = 123;
  const size_t N = 457;
  double min = -1.0;
  double max = 1.0;
  int seed = 19260817;

  MemRef<float, 2> output0({M, N}, 0);
  MemRef<float, 2> output1({M, N}, 0);

  runKernel(_mlir_ciface_kernel_autovec, min, max, seed, &output0);
  runKernel(_mlir_ciface_kernel_zuan, min, max, seed, &output1);

  output0.verify(output1, "Fill-Rng-2D", 0);

  for (size_t i = 0; i < 10; i++) {
    std::cout << "Index " << i << ":\tAutovec = " << std::setprecision(10)
              << output0[i] << "\tZuan = " << std::setprecision(10)
              << output1[i] << std::endl;
  }
}

BENCHMARK_CAPTURE(runBenchmark, zuan, _mlir_ciface_kernel_zuan)
    ->Unit(benchmark::kMicrosecond)
    ->Args({64, 56})
    ->Args({128, 128})
    ->Args({256, 256})
    ->Args({512, 512})
    ->Args({1024, 1024})
    ->Args({123, 457});

BENCHMARK_CAPTURE(runBenchmark, autovec, _mlir_ciface_kernel_autovec)
    ->Unit(benchmark::kMicrosecond)
    ->Args({64, 56})
    ->Args({128, 128})
    ->Args({256, 256})
    ->Args({512, 512})
    ->Args({1024, 1024})
    ->Args({123, 457});

int main(int argc, char **argv) {
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  std::cout << "------------------------------------------------" << std::endl;
  verifyFillRng2D();
  std::cout << "------------------------------------------------" << std::endl;
  return 0;
}

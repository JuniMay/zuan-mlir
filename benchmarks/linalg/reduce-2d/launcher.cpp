#include "benchmark/benchmark.h"
#include "common/MemRefUtils.h"
#include <cassert>
#include <iostream>
#include <random>

extern "C" {
void _mlir_ciface_kernel_autovec(MemRef<float, 2> *, MemRef<float, 1> *);
void _mlir_ciface_kernel_zuan(MemRef<float, 2> *, MemRef<float, 1> *);
}

using KernelFunc = void (*)(MemRef<float, 2> *, MemRef<float, 1> *);

static auto initializeData(size_t m, size_t n) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);

  MemRef<float, 2> tile({m, n}, 0);

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      tile[i * n + j] = dis(gen);
    }
  }

  return tile;
}

static void runKernel(KernelFunc kernel, MemRef<float, 2> *vec,
                      MemRef<float, 1> *output) {
  kernel(vec, output);
}

static void runBenchmark(benchmark::State &state, KernelFunc kernel) {
  size_t n = state.range(0);

  MemRef<float, 2> tile = initializeData(1, n);
  MemRef<float, 1> output({n}, 0);
  for (auto _ : state) {
    state.PauseTiming();
    output.fill(0);
    state.ResumeTiming();
    runKernel(kernel, &tile, &output);
  }
}

static void verifyReduce() {
  const size_t M = 1573;
  const size_t N = 1397;
  MemRef<float, 2> tile = initializeData(M, N);
  MemRef<float, 1> output0({N}, 0);
  MemRef<float, 1> output1({N}, 0);

  runKernel(_mlir_ciface_kernel_autovec, &tile, &output0);
  runKernel(_mlir_ciface_kernel_zuan, &tile, &output1);

  output0.verify(output1, "Reduce-2D", 0.0001);

  for (size_t i = 0; i < 10; i++) {
    std::cout << "Index " << i << ":\tAutovec=" << std::setprecision(10)
              << output0[i] << " Zuan=" << std::setprecision(10) << output1[i]
              << std::endl;
  }
}

BENCHMARK_CAPTURE(runBenchmark, zuan, _mlir_ciface_kernel_zuan)
    ->Unit(benchmark::kMillisecond)
    ->Args({1 << 10, 1 << 10})
    ->Args({1 << 12, 1 << 12})
    ->Args({1 << 14, 1 << 14})
    ->Args({1 << 16, 1 << 16})
    ->Args({1 << 18, 1 << 18})
    ->Args({1 << 20, 1 << 20})
    ->Args({1397319, 1397319});

BENCHMARK_CAPTURE(runBenchmark, autovec, _mlir_ciface_kernel_autovec)
    ->Unit(benchmark::kMillisecond)
    ->Args({1 << 10, 1 << 10})
    ->Args({1 << 12, 1 << 12})
    ->Args({1 << 14, 1 << 14})
    ->Args({1 << 16, 1 << 16})
    ->Args({1 << 18, 1 << 18})
    ->Args({1 << 20, 1 << 20})
    ->Args({1397319, 1397319});

int main(int argc, char **argv) {
  std::cout << "------------------------------------------------" << std::endl;
  verifyReduce();
  std::cout << "------------------------------------------------" << std::endl;
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}

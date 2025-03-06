#include "benchmark/benchmark.h"
#include "common/MemRefUtils.h"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <random>

extern "C" {
void _mlir_ciface_kernel_scalar(MemRef<float, 2> *, MemRef<float, 2> *,
                                MemRef<float, 2> *);
void _mlir_ciface_kernel_zuan(MemRef<float, 2> *, MemRef<float, 2> *,
                              MemRef<float, 2> *);
}

using KernelFunc = void (*)(MemRef<float, 2> *, MemRef<float, 2> *,
                            MemRef<float, 2> *);

static auto initializeData(size_t m, size_t n, size_t k) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);

  MemRef<float, 2> input1({m, k}, 0);
  MemRef<float, 2> input2({k, n}, 0);

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < k; j++) {
      input1[i * k + j] = dis(gen);
    }
  }

  for (size_t i = 0; i < k; i++) {
    for (size_t j = 0; j < n; j++) {
      input2[i * n + j] = dis(gen);
    }
  }

  return std::make_tuple(std::move(input1), std::move(input2));
}

static void runKernel(KernelFunc kernel, MemRef<float, 2> *input1,
                      MemRef<float, 2> *input2, MemRef<float, 2> *output) {
  kernel(input1, input2, output);
}

static void runBenchmark(benchmark::State &state, KernelFunc kernel) {
  size_t m = state.range(0);
  size_t n = state.range(1);
  size_t k = state.range(2);

  auto [input1, input2] = initializeData(m, n, k);
  MemRef<float, 2> output({m, n}, 0);
  for (auto _ : state) {
    state.PauseTiming();
    output.fill(0);
    state.ResumeTiming();
    runKernel(kernel, &input1, &input2, &output);
  }
}

static void verifyMatmul() {
  const size_t M = 511;
  const size_t N = 237;
  const size_t K = 123;

  auto [input1, input2] = initializeData(M, N, K);
  MemRef<float, 2> naive_buffer({M, N}, 0);
  MemRef<float, 2> zuan_buffer({M, N}, 0);

  runKernel(_mlir_ciface_kernel_zuan, &input1, &input2, &zuan_buffer);
  runKernel(_mlir_ciface_kernel_scalar, &input1, &input2, &naive_buffer);

  zuan_buffer.verify(naive_buffer, "Matmul", 0.0001);

  // print first 10 elements
  for (int i = 0; i < 10; i++) {
    std::cout << "Index " << i << ":\tNaive=" << std::setprecision(10)
              << naive_buffer[i] << " Zuan=" << std::setprecision(10)
              << zuan_buffer[i] << std::endl;
  }
}

BENCHMARK_CAPTURE(runBenchmark, zuan, _mlir_ciface_kernel_zuan)
    ->Unit(benchmark::kMillisecond)
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Args({511, 237, 123})
    ->Args({1023, 509, 2173});

BENCHMARK_CAPTURE(runBenchmark, naive, _mlir_ciface_kernel_scalar)
    ->Unit(benchmark::kMillisecond)
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Args({511, 237, 123})
    ->Args({1023, 509, 2173});

int main(int argc, char **argv) {
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  std::cout << "------------------------------------------------" << std::endl;
  verifyMatmul();
  std::cout << "------------------------------------------------" << std::endl;

  return 0;
}

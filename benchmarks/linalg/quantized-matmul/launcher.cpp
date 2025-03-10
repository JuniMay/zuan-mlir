#include "benchmark/benchmark.h"
#include "common/MemRefUtils.h"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>

extern "C" {
void _mlir_ciface_kernel_autovec_16(MemRef<int8_t, 2> *, MemRef<int8_t, 2> *,
                                    int, int, MemRef<int64_t, 2> *);
void _mlir_ciface_kernel_zuan_8_2(MemRef<int8_t, 2> *, MemRef<int8_t, 2> *, int,
                                  int, MemRef<int64_t, 2> *);
}

using KernelFunc = void (*)(MemRef<int8_t, 2> *, MemRef<int8_t, 2> *, int, int,
                            MemRef<int64_t, 2> *);

static auto initializeData(size_t m, size_t n, size_t k) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int8_t> dis(-128, 127);

  MemRef<int8_t, 2> input0({m, k}, 0);
  MemRef<int8_t, 2> input1({k, n}, 0);

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < k; j++) {
      input0[i * k + j] = dis(gen);
    }
  }

  for (size_t i = 0; i < k; i++) {
    for (size_t j = 0; j < n; j++) {
      input1[i * n + j] = dis(gen);
    }
  }

  return std::make_tuple(std::move(input0), std::move(input1));
}

static void runKernel(KernelFunc kernel, MemRef<int8_t, 2> *input0,
                      MemRef<int8_t, 2> *input1, int zp0, int zp1,
                      MemRef<int64_t, 2> *output) {
  kernel(input0, input1, zp0, zp1, output);
}

static void runBenchmark(benchmark::State &state, KernelFunc kernel) {
  size_t m = state.range(0);
  size_t n = state.range(1);
  size_t k = state.range(2);
  int zp0 = state.range(3);
  int zp1 = state.range(4);

  auto [input0, input1] = initializeData(m, n, k);
  MemRef<int64_t, 2> output({m, n}, 0);
  for (auto _ : state) {
    state.PauseTiming();
    output.fill(0);
    state.ResumeTiming();
    runKernel(kernel, &input0, &input1, zp0, zp1, &output);
  }
}

static void verifyMatmul() {
  const size_t M = 511;
  const size_t N = 237;
  const size_t K = 123;
  const int zp0 = 1;
  const int zp1 = 2;

  auto [input1, input2] = initializeData(M, N, K);
  MemRef<int64_t, 2> output0({M, N}, 0);
  MemRef<int64_t, 2> output1({M, N}, 0);

  runKernel(_mlir_ciface_kernel_autovec_16, &input1, &input2, zp0, zp1,
            &output0);
  runKernel(_mlir_ciface_kernel_zuan_8_2, &input1, &input2, zp0, zp1, &output1);

  output0.verify(output1, "quantized-matmul", 0);

  // print first 10 elements
  for (int i = 0; i < 10; i++) {
    std::cout << "Index " << i << ":\tAutovec=" << output0[i]
              << " Zuan=" << output1[i] << std::endl;
  }
}

BENCHMARK_CAPTURE(runBenchmark, zuan, _mlir_ciface_kernel_zuan_8_2)
    ->Unit(benchmark::kMillisecond)
    ->Args({64, 56, 32, 1, 2})
    ->Args({128, 128, 32, 1, 2})
    ->Args({256, 256, 32, 1, 2})
    ->Args({512, 512, 32, 1, 2})
    ->Args({1024, 1024, 32, 1, 2})
    ->Args({511, 237, 123, 1, 2})
    ->Args({1023, 509, 2173, 1, 2});

BENCHMARK_CAPTURE(runBenchmark, autovec, _mlir_ciface_kernel_autovec_16)
    ->Unit(benchmark::kMillisecond)
    ->Args({64, 56, 32, 1, 2})
    ->Args({128, 128, 32, 1, 2})
    ->Args({256, 256, 32, 1, 2})
    ->Args({512, 512, 32, 1, 2})
    ->Args({1024, 1024, 32, 1, 2})
    ->Args({511, 237, 123, 1, 2})
    ->Args({1023, 509, 2173, 1, 2});

int main(int argc, char **argv) {
  std::cout << "------------------------------------------------" << std::endl;
  verifyMatmul();
  std::cout << "------------------------------------------------" << std::endl;
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}

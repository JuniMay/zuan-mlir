#include "benchmark/benchmark.h"
#include "common/MemRefUtils.h"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <random>

extern "C" {
void _mlir_ciface_kernel_autovec_8(MemRef<float, 2> *, MemRef<float, 2> *,
                                   MemRef<float, 2> *);
void _mlir_ciface_kernel_autovec_16(MemRef<float, 2> *, MemRef<float, 2> *,
                                    MemRef<float, 2> *);
void _mlir_ciface_kernel_autovec_32(MemRef<float, 2> *, MemRef<float, 2> *,
                                    MemRef<float, 2> *);
void _mlir_ciface_kernel_autovec_64(MemRef<float, 2> *, MemRef<float, 2> *,
                                    MemRef<float, 2> *);
void _mlir_ciface_kernel_zuan_8_4(MemRef<float, 2> *, MemRef<float, 2> *,
                                  MemRef<float, 2> *);
void _mlir_ciface_kernel_zuan_16_2(MemRef<float, 2> *, MemRef<float, 2> *,
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
  MemRef<float, 2> autovec({M, N}, 0);
  runKernel(_mlir_ciface_kernel_autovec_16, &input1, &input2, &autovec);

  MemRef<float, 2> zuan_8_4({M, N}, 0);
  runKernel(_mlir_ciface_kernel_zuan_8_4, &input1, &input2, &zuan_8_4);

  MemRef<float, 2> zuan_16_2({M, N}, 0);
  runKernel(_mlir_ciface_kernel_zuan_16_2, &input1, &input2, &zuan_16_2);

  autovec.verify(zuan_16_2, "Matmul", 0.0001);
  autovec.verify(zuan_8_4, "Matmul", 0.0001);

  // print first 10 elements
  for (int i = 0; i < 10; i++) {
    std::cerr << "Index " << i << ":\tAutovec=" << std::setprecision(10)
              << autovec[i] << " Zuan-8-4=" << std::setprecision(10)
              << zuan_8_4[i] << " Zuan-16-2=" << std::setprecision(10)
              << zuan_16_2[i] << std::endl;
  }
}

BENCHMARK_CAPTURE(runBenchmark, zuan_8_4, _mlir_ciface_kernel_zuan_8_4)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({
        {64, 128, 192, 256, 384, 512, 768, 1024},
        {64, 128, 192, 256, 384, 512, 768, 1024},
        {64, 128, 192, 256, 384, 512, 768, 1024},
    });
BENCHMARK_CAPTURE(runBenchmark, zuan_16_2, _mlir_ciface_kernel_zuan_16_2)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({
        {64, 128, 192, 256, 384, 512, 768, 1024},
        {64, 128, 192, 256, 384, 512, 768, 1024},
        {64, 128, 192, 256, 384, 512, 768, 1024},
    });

BENCHMARK_CAPTURE(runBenchmark, autovec_8, _mlir_ciface_kernel_autovec_8)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({
        {64, 128, 192, 256, 384, 512, 768, 1024},
        {64, 128, 192, 256, 384, 512, 768, 1024},
        {64, 128, 192, 256, 384, 512, 768, 1024},
    });
BENCHMARK_CAPTURE(runBenchmark, autovec_16, _mlir_ciface_kernel_autovec_16)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({
        {64, 128, 192, 256, 384, 512, 768, 1024},
        {64, 128, 192, 256, 384, 512, 768, 1024},
        {64, 128, 192, 256, 384, 512, 768, 1024},
    });
BENCHMARK_CAPTURE(runBenchmark, autovec_32, _mlir_ciface_kernel_autovec_32)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({
        {64, 128, 192, 256, 384, 512, 768, 1024},
        {64, 128, 192, 256, 384, 512, 768, 1024},
        {64, 128, 192, 256, 384, 512, 768, 1024},
    });
BENCHMARK_CAPTURE(runBenchmark, autovec_64, _mlir_ciface_kernel_autovec_64)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({
        {64, 128, 192, 256, 384, 512, 768, 1024},
        {64, 128, 192, 256, 384, 512, 768, 1024},
        {64, 128, 192, 256, 384, 512, 768, 1024},
    });

int main(int argc, char **argv) {
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  std::cerr << "------------------------------------------------" << std::endl;
  verifyMatmul();
  std::cerr << "------------------------------------------------" << std::endl;

  return 0;
}

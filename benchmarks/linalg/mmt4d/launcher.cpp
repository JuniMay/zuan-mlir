#include "benchmark/benchmark.h"
#include "common/MemRefUtils.h"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <random>

extern "C" {
void _mlir_ciface_kernel_autovec_8(MemRef<float, 4> *, MemRef<float, 4> *,
                                   MemRef<float, 4> *);
void _mlir_ciface_kernel_autovec_16(MemRef<float, 4> *, MemRef<float, 4> *,
                                    MemRef<float, 4> *);
void _mlir_ciface_kernel_autovec_32(MemRef<float, 4> *, MemRef<float, 4> *,
                                    MemRef<float, 4> *);
void _mlir_ciface_kernel_autovec_64(MemRef<float, 4> *, MemRef<float, 4> *,
                                    MemRef<float, 4> *);
void _mlir_ciface_kernel_zuan_8_4(MemRef<float, 4> *, MemRef<float, 4> *,
                                  MemRef<float, 4> *);
void _mlir_ciface_kernel_zuan_16_2(MemRef<float, 4> *, MemRef<float, 4> *,
                                   MemRef<float, 4> *);
}

using KernelFunc = void (*)(MemRef<float, 4> *, MemRef<float, 4> *,
                            MemRef<float, 4> *);

static auto initializeData(size_t m, size_t n, size_t k, size_t m0, size_t n0,
                           size_t k0) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);

  MemRef<float, 4> input1({m, k, m0, k0}, 0);
  MemRef<float, 4> input2({n, k, n0, k0}, 0);

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < k; j++) {
      //   input1[i * k + j] = dis(gen);
      for (size_t i0 = 0; i0 < m0; i0++) {
        for (size_t j0 = 0; j0 < k0; j0++) {
          input1[i * k * m0 * k0 + j * m0 * k0 + i0 * k0 + j0] = dis(gen);
        }
      }
    }
  }

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < k; j++) {
      for (size_t i0 = 0; i0 < n0; i0++) {
        for (size_t j0 = 0; j0 < k0; j0++) {
          input2[i * k * n0 * k0 + j * n0 * k0 + i0 * k0 + j0] = dis(gen);
        }
      }
    }
  }

  return std::make_tuple(std::move(input1), std::move(input2));
}

static void runKernel(KernelFunc kernel, MemRef<float, 4> *input1,
                      MemRef<float, 4> *input2, MemRef<float, 4> *output) {
  kernel(input1, input2, output);
}

static void runBenchmark(benchmark::State &state, KernelFunc kernel) {
  size_t m = state.range(0);
  size_t n = state.range(1);
  size_t k = state.range(2);
  size_t m0 = state.range(3);
  size_t n0 = state.range(4);
  size_t k0 = state.range(5);

  auto [input1, input2] = initializeData(m, n, k, m0, n0, k0);
  MemRef<float, 4> output({m, n, m0, n0}, 0);
  for (auto _ : state) {
    state.PauseTiming();
    output.fill(0);
    state.ResumeTiming();
    runKernel(kernel, &input1, &input2, &output);
  }
}

static void verifyMmt4D() {
  const size_t M = 3;
  const size_t N = 7;
  const size_t K = 9;
  const size_t M0 = 51;
  const size_t N0 = 23;
  const size_t K0 = 13;

  auto [input1, input2] = initializeData(M, N, K, M0, N0, K0);
  MemRef<float, 4> autovec({M, N, M0, N0}, 0);
  runKernel(_mlir_ciface_kernel_autovec_16, &input1, &input2, &autovec);

  MemRef<float, 4> zuan_8_4({M, N, M0, N0}, 0);
  runKernel(_mlir_ciface_kernel_zuan_8_4, &input1, &input2, &zuan_8_4);
  MemRef<float, 4> zuan_16_2({M, N, M0, N0}, 0);
  runKernel(_mlir_ciface_kernel_zuan_16_2, &input1, &input2, &zuan_16_2);

  autovec.verify(zuan_16_2, "mmt4d", 0.0001);

  // print first 10 elements
  for (int i = 0; i < 10; i++) {
    std::cout << "Index " << i << ":\tAutovec=" << std::setprecision(10)
              << autovec[i] << " Zuan-16-2=" << std::setprecision(10)
              << zuan_16_2[i] << "Zuan-8-4=" << std::setprecision(10)
              << zuan_8_4[i] << std::endl;
  }
}

BENCHMARK_CAPTURE(runBenchmark, zuan_8_4, _mlir_ciface_kernel_zuan_8_4)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({
        {3, 4, 5, 6, 7, 8},
        {3, 4, 5, 6, 7, 8},
        {3, 4, 5, 6, 7, 8},
        {16, 32, 51, 64, 128},
        {16, 32, 51, 64, 128},
        {16, 32, 51, 64, 128},
    });
BENCHMARK_CAPTURE(runBenchmark, zuan_16_2, _mlir_ciface_kernel_zuan_16_2)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({
        {3, 4, 5, 6, 7, 8},
        {3, 4, 5, 6, 7, 8},
        {3, 4, 5, 6, 7, 8},
        {16, 32, 51, 64, 128},
        {16, 32, 51, 64, 128},
        {16, 32, 51, 64, 128},
    });

BENCHMARK_CAPTURE(runBenchmark, autovec_8, _mlir_ciface_kernel_autovec_8)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({
        {3, 4, 5, 6, 7, 8},
        {3, 4, 5, 6, 7, 8},
        {3, 4, 5, 6, 7, 8},
        {16, 32, 51, 64, 128},
        {16, 32, 51, 64, 128},
        {16, 32, 51, 64, 128},
    });
BENCHMARK_CAPTURE(runBenchmark, autovec_16, _mlir_ciface_kernel_autovec_16)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({
        {3, 4, 5, 6, 7, 8},
        {3, 4, 5, 6, 7, 8},
        {3, 4, 5, 6, 7, 8},
        {16, 32, 51, 64, 128},
        {16, 32, 51, 64, 128},
        {16, 32, 51, 64, 128},
    });
BENCHMARK_CAPTURE(runBenchmark, autovec_32, _mlir_ciface_kernel_autovec_32)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({
        {3, 4, 5, 6, 7, 8},
        {3, 4, 5, 6, 7, 8},
        {3, 4, 5, 6, 7, 8},
        {16, 32, 51, 64, 128},
        {16, 32, 51, 64, 128},
        {16, 32, 51, 64, 128},
    });
BENCHMARK_CAPTURE(runBenchmark, autovec_64, _mlir_ciface_kernel_autovec_64)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({
        {3, 4, 5, 6, 7, 8},
        {3, 4, 5, 6, 7, 8},
        {3, 4, 5, 6, 7, 8},
        {16, 32, 51, 64, 128},
        {16, 32, 51, 64, 128},
        {16, 32, 51, 64, 128},
    });

int main(int argc, char **argv) {
  std::cout << "------------------------------------------------" << std::endl;
  verifyMmt4D();
  std::cout << "------------------------------------------------" << std::endl;

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  return 0;
}

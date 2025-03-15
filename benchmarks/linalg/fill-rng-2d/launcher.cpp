#include "benchmark/benchmark.h"
#include "common/MemRefUtils.h"
#include <cassert>
#include <iomanip>
#include <iostream>

extern "C" {
void _mlir_ciface_kernel_autovec_8(double, double, int, MemRef<float, 2> *);
void _mlir_ciface_kernel_autovec_16(double, double, int, MemRef<float, 2> *);
void _mlir_ciface_kernel_autovec_32(double, double, int, MemRef<float, 2> *);
void _mlir_ciface_kernel_autovec_64(double, double, int, MemRef<float, 2> *);

void _mlir_ciface_kernel_zuan_4_1(double, double, int, MemRef<float, 2> *);
void _mlir_ciface_kernel_zuan_4_2(double, double, int, MemRef<float, 2> *);
void _mlir_ciface_kernel_zuan_4_4(double, double, int, MemRef<float, 2> *);
void _mlir_ciface_kernel_zuan_8_1(double, double, int, MemRef<float, 2> *);
void _mlir_ciface_kernel_zuan_8_2(double, double, int, MemRef<float, 2> *);

void _mlir_ciface_kernel_transform_4_1(double, double, int, MemRef<float, 2> *);
void _mlir_ciface_kernel_transform_8_1(double, double, int, MemRef<float, 2> *);
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

  MemRef<float, 2> autovec({M, N}, 0);

  runKernel(_mlir_ciface_kernel_autovec_16, min, max, seed, &autovec);
  MemRef<float, 2> zuan_8_1({M, N}, 0);
  runKernel(_mlir_ciface_kernel_zuan_8_1, min, max, seed, &zuan_8_1);

  MemRef<float, 2> zuan_8_2({M, N}, 0);
  runKernel(_mlir_ciface_kernel_zuan_8_2, min, max, seed, &zuan_8_2);
  MemRef<float, 2> zuan_4_1({M, N}, 0);
  runKernel(_mlir_ciface_kernel_zuan_4_1, min, max, seed, &zuan_4_1);
  MemRef<float, 2> zuan_4_2({M, N}, 0);
  runKernel(_mlir_ciface_kernel_zuan_4_2, min, max, seed, &zuan_4_2);
  MemRef<float, 2> zuan_4_4({M, N}, 0);
  runKernel(_mlir_ciface_kernel_zuan_4_4, min, max, seed, &zuan_4_4);

  autovec.verify(zuan_8_1, "Fill-Rng-2D-Zuan-8-1", 0);
  autovec.verify(zuan_8_2, "Fill-Rng-2D-Zuan-8-2", 0);
  autovec.verify(zuan_4_1, "Fill-Rng-2D-Zuan-4-1", 0);
  autovec.verify(zuan_4_2, "Fill-Rng-2D-Zuan-4-2", 0);
  autovec.verify(zuan_4_4, "Fill-Rng-2D-Zuan-4-4", 0);

  for (size_t i = 0; i < 10; i++) {
    std::cerr << "Index " << i << ":\tAutovec = " << std::setprecision(10)
              << autovec[i] << "\tZuan-8-2 = " << std::setprecision(10)
              << zuan_8_2[i] << "\tZuan-8-1 = " << std::setprecision(10)
              << zuan_8_1[i] << "\tZuan-4-1 = " << std::setprecision(10)
              << zuan_4_1[i] << "\tZuan-4-2 = " << std::setprecision(10)
              << zuan_4_2[i] << "\tZuan-4-4 = " << std::setprecision(10)
              << zuan_4_4[i] << std::endl;
  }
}

//-------------------------------------------------------------------
// Zuan
//-------------------------------------------------------------------

BENCHMARK_CAPTURE(runBenchmark, zuan_8_2, _mlir_ciface_kernel_zuan_8_2)
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({{256, 512}, {256, 384, 512, 1024, 2048, 4096}});

BENCHMARK_CAPTURE(runBenchmark, zuan_8_1, _mlir_ciface_kernel_zuan_8_1)
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({{256, 512}, {256, 384, 512, 1024, 2048, 4096}});
BENCHMARK_CAPTURE(runBenchmark, zuan_4_1, _mlir_ciface_kernel_zuan_4_1)
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({{256, 512}, {256, 384, 512, 1024, 2048, 4096}});

BENCHMARK_CAPTURE(runBenchmark, zuan_4_2, _mlir_ciface_kernel_zuan_4_2)
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({{256, 512}, {256, 384, 512, 1024, 2048, 4096}});
BENCHMARK_CAPTURE(runBenchmark, zuan_4_4, _mlir_ciface_kernel_zuan_4_4)
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({{256, 512}, {256, 384, 512, 1024, 2048, 4096}});

//-------------------------------------------------------------------
// Transform Dialect
//-------------------------------------------------------------------

BENCHMARK_CAPTURE(runBenchmark, transform_8_1, _mlir_ciface_kernel_transform_8_1)
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({{256, 512}, {256, 384, 512, 1024, 2048, 4096}});
BENCHMARK_CAPTURE(runBenchmark, transform_4_1, _mlir_ciface_kernel_transform_4_1)
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({{256, 512}, {256, 384, 512, 1024, 2048, 4096}});

//-------------------------------------------------------------------
// Auto-vectorization
//-------------------------------------------------------------------

BENCHMARK_CAPTURE(runBenchmark, autovec_8, _mlir_ciface_kernel_autovec_8)
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({{256, 512}, {256, 384, 512, 1024, 2048, 4096}});
BENCHMARK_CAPTURE(runBenchmark, autovec_16, _mlir_ciface_kernel_autovec_16)
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({{256, 512}, {256, 384, 512, 1024, 2048, 4096}});
BENCHMARK_CAPTURE(runBenchmark, autovec_32, _mlir_ciface_kernel_autovec_32)
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({{256, 512}, {256, 384, 512, 1024, 2048, 4096}});

BENCHMARK_CAPTURE(runBenchmark, autovec_64, _mlir_ciface_kernel_autovec_64)
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({{256, 512}, {256, 384, 512, 1024, 2048, 4096}});

int main(int argc, char **argv) {
  std::cerr << "------------------------------------------------" << std::endl;
  verifyFillRng2D();
  std::cerr << "------------------------------------------------" << std::endl;
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}

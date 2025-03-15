#include "benchmark/benchmark.h"
#include "common/MemRefUtils.h"
#include <cassert>
#include <iostream>
#include <random>

extern "C" {
void _mlir_ciface_kernel_autovec_8(MemRef<float, 2> *, MemRef<float, 1> *);
void _mlir_ciface_kernel_autovec_16(MemRef<float, 2> *, MemRef<float, 1> *);
void _mlir_ciface_kernel_autovec_32(MemRef<float, 2> *, MemRef<float, 1> *);
void _mlir_ciface_kernel_autovec_64(MemRef<float, 2> *, MemRef<float, 1> *);

void _mlir_ciface_kernel_zuan_8_2(MemRef<float, 2> *, MemRef<float, 1> *);
void _mlir_ciface_kernel_zuan_8_4(MemRef<float, 2> *, MemRef<float, 1> *);
void _mlir_ciface_kernel_zuan_8_8(MemRef<float, 2> *, MemRef<float, 1> *);
void _mlir_ciface_kernel_zuan_16_1(MemRef<float, 2> *, MemRef<float, 1> *);
void _mlir_ciface_kernel_zuan_16_2(MemRef<float, 2> *, MemRef<float, 1> *);
void _mlir_ciface_kernel_zuan_16_4(MemRef<float, 2> *, MemRef<float, 1> *);

void _mlir_ciface_kernel_transform_8_2(MemRef<float, 2> *, MemRef<float, 1> *);
void _mlir_ciface_kernel_transform_8_4(MemRef<float, 2> *, MemRef<float, 1> *);
void _mlir_ciface_kernel_transform_8_8(MemRef<float, 2> *, MemRef<float, 1> *);
void _mlir_ciface_kernel_transform_16_1(MemRef<float, 2> *, MemRef<float, 1> *);
void _mlir_ciface_kernel_transform_16_2(MemRef<float, 2> *, MemRef<float, 1> *);
void _mlir_ciface_kernel_transform_16_4(MemRef<float, 2> *, MemRef<float, 1> *);
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
  MemRef<float, 1> autovec({N}, 0);

  runKernel(_mlir_ciface_kernel_autovec_16, &tile, &autovec);

  MemRef<float, 1> zuan_8_2({N}, 0);
  runKernel(_mlir_ciface_kernel_zuan_8_2, &tile, &zuan_8_2);
  MemRef<float, 1> zuan_8_4({N}, 0);
  runKernel(_mlir_ciface_kernel_zuan_8_4, &tile, &zuan_8_4);
  MemRef<float, 1> zuan_8_8({N}, 0);
  runKernel(_mlir_ciface_kernel_zuan_8_8, &tile, &zuan_8_8);
  MemRef<float, 1> zuan_16_1({N}, 0);
  runKernel(_mlir_ciface_kernel_zuan_16_1, &tile, &zuan_16_1);
  MemRef<float, 1> zuan_16_2({N}, 0);
  runKernel(_mlir_ciface_kernel_zuan_16_2, &tile, &zuan_16_2);
  MemRef<float, 1> zuan_16_4({N}, 0);
  runKernel(_mlir_ciface_kernel_zuan_16_4, &tile, &zuan_16_4);

  // autovec.verify(zuan_16_2, "Reduce-2D", 0.0001);
  autovec.verify(zuan_8_2, "Reduce-2D-Zuan-8-2", 0.0001);
  autovec.verify(zuan_8_4, "Reduce-2D-Zuan-8-4", 0.0001);
  autovec.verify(zuan_8_8, "Reduce-2D-Zuan-8-8", 0.0001);
  autovec.verify(zuan_16_1, "Reduce-2D-Zuan-16-1", 0.0001);
  autovec.verify(zuan_16_2, "Reduce-2D-Zuan-16-2", 0.0001);
  autovec.verify(zuan_16_4, "Reduce-2D-Zuan-16-4", 0.0001);

  for (size_t i = 0; i < 10; i++) {
    std::cerr << "Index " << i << ":\tAutovec=" << std::setprecision(10)
              << autovec[i] << " Zuan-8-2=" << std::setprecision(10)
              << zuan_8_2[i] << " Zuan-8-4=" << std::setprecision(10)
              << zuan_8_4[i] << " Zuan-8-8=" << std::setprecision(10)
              << zuan_8_8[i] << " Zuan-16-1=" << std::setprecision(10)
              << zuan_16_1[i] << " Zuan-16-2=" << std::setprecision(10)
              << zuan_16_2[i] << " Zuan-16-4=" << std::setprecision(10)
              << zuan_16_4[i] << std::endl;
  }
}

//-------------------------------------------------------------------
// Zuan
//-------------------------------------------------------------------

BENCHMARK_CAPTURE(runBenchmark, zuan_8_2, _mlir_ciface_kernel_zuan_8_2)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{512, 1 << 10, 1 << 12, 1 << 14},
                   {1 << 16, 1 << 18, 1 << 22}});
BENCHMARK_CAPTURE(runBenchmark, zuan_8_4, _mlir_ciface_kernel_zuan_8_4)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{512, 1 << 10, 1 << 12, 1 << 14},
                   {1 << 16, 1 << 18, 1 << 22}});
BENCHMARK_CAPTURE(runBenchmark, zuan_8_8, _mlir_ciface_kernel_zuan_8_8)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{512, 1 << 10, 1 << 12, 1 << 14},
                   {1 << 16, 1 << 18, 1 << 22}});
BENCHMARK_CAPTURE(runBenchmark, zuan_16_1, _mlir_ciface_kernel_zuan_16_1)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{512, 1 << 10, 1 << 12, 1 << 14},
                   {1 << 16, 1 << 18, 1 << 22}});
BENCHMARK_CAPTURE(runBenchmark, zuan_16_2, _mlir_ciface_kernel_zuan_16_2)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{512, 1 << 10, 1 << 12, 1 << 14},
                   {1 << 16, 1 << 18, 1 << 22}});
BENCHMARK_CAPTURE(runBenchmark, zuan_16_4, _mlir_ciface_kernel_zuan_16_4)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{512, 1 << 10, 1 << 12, 1 << 14},
                   {1 << 16, 1 << 18, 1 << 22}});

//-------------------------------------------------------------------
// Transform Dialect
//-------------------------------------------------------------------

BENCHMARK_CAPTURE(runBenchmark, transform_8_2,
                  _mlir_ciface_kernel_transform_8_2)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{512, 1 << 10, 1 << 12, 1 << 14},
                   {1 << 16, 1 << 18, 1 << 22}});
BENCHMARK_CAPTURE(runBenchmark, transform_8_4,
                  _mlir_ciface_kernel_transform_8_4)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{512, 1 << 10, 1 << 12, 1 << 14},
                   {1 << 16, 1 << 18, 1 << 22}});
BENCHMARK_CAPTURE(runBenchmark, transform_8_8,
                  _mlir_ciface_kernel_transform_8_8)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{512, 1 << 10, 1 << 12, 1 << 14},
                   {1 << 16, 1 << 18, 1 << 22}});
BENCHMARK_CAPTURE(runBenchmark, transform_16_1,
                  _mlir_ciface_kernel_transform_16_1)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{512, 1 << 10, 1 << 12, 1 << 14},
                   {1 << 16, 1 << 18, 1 << 22}});
BENCHMARK_CAPTURE(runBenchmark, transform_16_2,
                  _mlir_ciface_kernel_transform_16_2)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{512, 1 << 10, 1 << 12, 1 << 14},
                   {1 << 16, 1 << 18, 1 << 22}});
BENCHMARK_CAPTURE(runBenchmark, transform_16_4,
                  _mlir_ciface_kernel_transform_16_4)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{512, 1 << 10, 1 << 12, 1 << 14},
                   {1 << 16, 1 << 18, 1 << 22}});

//-------------------------------------------------------------------
// Auto-vectorization
//-------------------------------------------------------------------

BENCHMARK_CAPTURE(runBenchmark, autovec_8, _mlir_ciface_kernel_autovec_8)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{512, 1 << 10, 1 << 12, 1 << 14},
                   {1 << 16, 1 << 18, 1 << 22}});

BENCHMARK_CAPTURE(runBenchmark, autovec_16, _mlir_ciface_kernel_autovec_16)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{512, 1 << 10, 1 << 12, 1 << 14},
                   {1 << 16, 1 << 18, 1 << 22}});

BENCHMARK_CAPTURE(runBenchmark, autovec_32, _mlir_ciface_kernel_autovec_32)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{512, 1 << 10, 1 << 12, 1 << 14},
                   {1 << 16, 1 << 18, 1 << 22}});

BENCHMARK_CAPTURE(runBenchmark, autovec_64, _mlir_ciface_kernel_autovec_64)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{512, 1 << 10, 1 << 12, 1 << 14},
                   {1 << 16, 1 << 18, 1 << 22}});

int main(int argc, char **argv) {
  std::cerr << "------------------------------------------------" << std::endl;
  verifyReduce();
  std::cerr << "------------------------------------------------" << std::endl;
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}

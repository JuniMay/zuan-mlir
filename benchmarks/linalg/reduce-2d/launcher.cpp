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

  autovec.verify(zuan_8_2, "reduce-2d-zuan-8-2", 0.0001);
  autovec.verify(zuan_8_4, "reduce-2d-zuan-8-4", 0.0001);
  autovec.verify(zuan_8_8, "reduce-2d-zuan-8-8", 0.0001);
  autovec.verify(zuan_16_1, "reduce-2d-zuan-16-1", 0.0001);
  autovec.verify(zuan_16_2, "reduce-2d-zuan-16-2", 0.0001);
  autovec.verify(zuan_16_4, "reduce-2d-zuan-16-4", 0.0001);

  MemRef<float, 1> transform_8_2({N}, 0);
  runKernel(_mlir_ciface_kernel_transform_8_2, &tile, &transform_8_2);
  MemRef<float, 1> transform_8_4({N}, 0);
  runKernel(_mlir_ciface_kernel_transform_8_4, &tile, &transform_8_4);
  MemRef<float, 1> transform_8_8({N}, 0);
  runKernel(_mlir_ciface_kernel_transform_8_8, &tile, &transform_8_8);
  MemRef<float, 1> transform_16_1({N}, 0);
  runKernel(_mlir_ciface_kernel_transform_16_1, &tile, &transform_16_1);
  MemRef<float, 1> transform_16_2({N}, 0);
  runKernel(_mlir_ciface_kernel_transform_16_2, &tile, &transform_16_2);
  MemRef<float, 1> transform_16_4({N}, 0);
  runKernel(_mlir_ciface_kernel_transform_16_4, &tile, &transform_16_4);

  autovec.verify(transform_8_2, "reduce-2d-transform-8-2", 0.0001);
  autovec.verify(transform_8_4, "reduce-2d-transform-8-4", 0.0001);
  autovec.verify(transform_8_8, "reduce-2d-transform-8-8", 0.0001);
  autovec.verify(transform_16_1, "reduce-2d-transform-16-1", 0.0001);
  autovec.verify(transform_16_2, "reduce-2d-transform-16-2", 0.0001);
  autovec.verify(transform_16_4, "reduce-2d-transform-16-4", 0.0001);

  for (size_t i = 0; i < 10; i++) {
    std::cerr << "Index " << i << std::setprecision(10)
              << ": autovec=" << autovec[i]

              << "\tzuan-8-2=" << zuan_8_2[i] << "\tzuan-8-4=" << zuan_8_4[i]
              << "\tzuan-8-8=" << zuan_8_8[i] << "\tzuan-16-1=" << zuan_16_1[i]
              << "\tzuan-16-2=" << zuan_16_2[i]
              << "\tzuan-16-4=" << zuan_16_4[i]

              << "\ttransform-8-2=" << transform_8_2[i]
              << "\ttransform-8-4=" << transform_8_4[i]
              << "\ttransform-8-8=" << transform_8_8[i]
              << "\ttransform-16-1=" << transform_16_1[i]
              << "\ttransform-16-2=" << transform_16_2[i]
              << "\ttransform-16-4=" << transform_16_4[i]

              << std::endl;
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

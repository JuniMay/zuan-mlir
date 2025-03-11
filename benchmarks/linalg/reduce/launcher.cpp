#include "benchmark/benchmark.h"
#include "common/MemRefUtils.h"
#include <cassert>
#include <iostream>
#include <random>

extern "C" {
void _mlir_ciface_kernel_autovec_8(MemRef<float, 1> *, MemRef<float, 0> *);
void _mlir_ciface_kernel_autovec_16(MemRef<float, 1> *, MemRef<float, 0> *);
void _mlir_ciface_kernel_autovec_32(MemRef<float, 1> *, MemRef<float, 0> *);
void _mlir_ciface_kernel_autovec_64(MemRef<float, 1> *, MemRef<float, 0> *);
void _mlir_ciface_kernel_zuan_16_2(MemRef<float, 1> *, MemRef<float, 0> *);
}

using KernelFunc = void (*)(MemRef<float, 1> *, MemRef<float, 0> *);

static auto initializeData(size_t n) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);

  MemRef<float, 1> vec({n}, 0);

  for (size_t i = 0; i < n; i++) {
    vec[i] = dis(gen);
  }

  return vec;
}

static void runKernel(KernelFunc kernel, MemRef<float, 1> *vec,
                      MemRef<float, 0> *output) {
  kernel(vec, output);
}

static void runBenchmark(benchmark::State &state, KernelFunc kernel) {
  size_t n = state.range(0);

  MemRef<float, 1> vec = initializeData(n);
  MemRef<float, 0> output({}, 0);
  for (auto _ : state) {
    state.PauseTiming();
    output.fill(0);
    state.ResumeTiming();
    runKernel(kernel, &vec, &output);
  }
}

static void verifyReduce() {
  const size_t N = 1397;
  MemRef<float, 1> vec = initializeData(N);
  MemRef<float, 0> autovec({}, 0);
  MemRef<float, 0> zuan_16_2({}, 0);

  runKernel(_mlir_ciface_kernel_autovec_16, &vec, &autovec);
  runKernel(_mlir_ciface_kernel_zuan_16_2, &vec, &zuan_16_2);

  std::cerr << "Autovec:   " << autovec[0] << "\n";
  std::cerr << "Zuan-16-2: " << zuan_16_2[0] << "\n";
}

BENCHMARK_CAPTURE(runBenchmark, zuan_16_2, _mlir_ciface_kernel_zuan_16_2)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 22);

BENCHMARK_CAPTURE(runBenchmark, autovec_8, _mlir_ciface_kernel_autovec_8)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 22);
BENCHMARK_CAPTURE(runBenchmark, autovec_16, _mlir_ciface_kernel_autovec_16)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 22);
BENCHMARK_CAPTURE(runBenchmark, autovec_32, _mlir_ciface_kernel_autovec_32)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 22);
BENCHMARK_CAPTURE(runBenchmark, autovec_64, _mlir_ciface_kernel_autovec_64)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 22);

int main(int argc, char **argv) {
  std::cerr << "------------------------------------------------" << std::endl;
  verifyReduce();
  std::cerr << "------------------------------------------------" << std::endl;
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}

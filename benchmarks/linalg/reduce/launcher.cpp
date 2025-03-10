#include "benchmark/benchmark.h"
#include "common/MemRefUtils.h"
#include <cassert>
#include <iostream>
#include <random>

extern "C" {
void _mlir_ciface_kernel_autovec_16(MemRef<float, 1> *, MemRef<float, 0> *);
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
  MemRef<float, 0> output0({}, 0);
  MemRef<float, 0> output1({}, 0);

  runKernel(_mlir_ciface_kernel_autovec_16, &vec, &output0);
  runKernel(_mlir_ciface_kernel_zuan_16_2, &vec, &output1);

  std::cout << "Output0: " << output0[0] << "\n";
  std::cout << "Output1: " << output1[0] << "\n";
}

BENCHMARK_CAPTURE(runBenchmark, zuan, _mlir_ciface_kernel_zuan_16_2)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1 << 10)
    ->Arg(1 << 12)
    ->Arg(1 << 14)
    ->Arg(1 << 16)
    ->Arg(1 << 18)
    ->Arg(1 << 20)
    ->Arg(1397319);

BENCHMARK_CAPTURE(runBenchmark, autovec, _mlir_ciface_kernel_autovec_16)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1 << 10)
    ->Arg(1 << 12)
    ->Arg(1 << 14)
    ->Arg(1 << 16)
    ->Arg(1 << 18)
    ->Arg(1 << 20)
    ->Arg(1397319);

int main(int argc, char **argv) {
  std::cout << "------------------------------------------------" << std::endl;
  verifyReduce();
  std::cout << "------------------------------------------------" << std::endl;
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}

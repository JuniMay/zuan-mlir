#include "benchmark/benchmark.h"
#include "common/MemRefUtils.h"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <random>

extern "C" {
void _mlir_ciface_kernel_autovec(MemRef<_Float16, 1> *, MemRef<_Float16, 1> *,
                                 MemRef<float, 0> *);
void _mlir_ciface_kernel_zuan(MemRef<_Float16, 1> *, MemRef<_Float16, 1> *,
                              MemRef<float, 0> *);
}

using KernelFunc = void (*)(MemRef<_Float16, 1> *, MemRef<_Float16, 1> *,
                            MemRef<float, 0> *);

static auto initializeData(size_t n) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);

  MemRef<_Float16, 1> vec_a({n}, 0);
  MemRef<_Float16, 1> vec_b({n}, 0);

  for (size_t i = 0; i < n; i++) {
    vec_a[i] = dis(gen);
    vec_b[i] = dis(gen);
  }

  return std::make_tuple(std::move(vec_a), std::move(vec_b));
}

static void runKernel(KernelFunc kernel, MemRef<_Float16, 1> *vec_a,
                      MemRef<_Float16, 1> *vec_b, MemRef<float, 0> *output) {
  kernel(vec_a, vec_b, output);
}

static void runBenchmark(benchmark::State &state, KernelFunc kernel) {
  size_t n = state.range(0);

  auto [vec_a, vec_b] = initializeData(n);
  MemRef<float, 0> output({}, 0);
  for (auto _ : state) {
    state.PauseTiming();
    output.fill(0);
    state.ResumeTiming();
    runKernel(kernel, &vec_a, &vec_b, &output);
  }
}

static void verifyDotFp16() {
  const size_t N = 1397;
  auto [vec_a, vec_b] = initializeData(N);
  MemRef<float, 0> output0({}, 0);
  MemRef<float, 0> output1({}, 0);

  runKernel(_mlir_ciface_kernel_autovec, &vec_a, &vec_b, &output0);
  runKernel(_mlir_ciface_kernel_zuan, &vec_a, &vec_b, &output1);

  // The clang auto-vectorization generates ordered reduce, while in zuan it
  // uses unordered reduce. The difference in the order of reduction can lead to
  // different results.
  output0.verify(output1, "Dot-fp16", 10);

  std::cout << "Autovec = " << std::setprecision(10) << output0[0]
            << "\tZuan = " << std::setprecision(10) << output1[0] << std::endl;
}

BENCHMARK_CAPTURE(runBenchmark, autovec, _mlir_ciface_kernel_autovec)
    ->Unit(benchmark::kMillisecond)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Arg(65536)
    ->Arg(1397);

BENCHMARK_CAPTURE(runBenchmark, zuan, _mlir_ciface_kernel_zuan)
    ->Unit(benchmark::kMillisecond)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Arg(65536)
    ->Arg(1397);

int main(int argc, char **argv) {
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  std::cout << "------------------------------------------------" << std::endl;
  verifyDotFp16();
  std::cout << "------------------------------------------------" << std::endl;
  return 0;
}
#include "benchmark/benchmark.h"
#include "common/MemRefUtils.h"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <random>

extern "C" {

void _mlir_ciface_kernel_autovec_8(MemRef<_Float16, 1> *, MemRef<_Float16, 1> *,
                                   MemRef<float, 0> *);
void _mlir_ciface_kernel_autovec_16(MemRef<_Float16, 1> *,
                                    MemRef<_Float16, 1> *, MemRef<float, 0> *);
void _mlir_ciface_kernel_autovec_32(MemRef<_Float16, 1> *,
                                    MemRef<_Float16, 1> *, MemRef<float, 0> *);
void _mlir_ciface_kernel_autovec_64(MemRef<_Float16, 1> *,
                                    MemRef<_Float16, 1> *, MemRef<float, 0> *);

void _mlir_ciface_kernel_zuan_16_2(MemRef<_Float16, 1> *, MemRef<_Float16, 1> *,
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
  runKernel(_mlir_ciface_kernel_autovec_16, &vec_a, &vec_b, &output0);

  MemRef<float, 0> output_16_2({}, 0);
  runKernel(_mlir_ciface_kernel_zuan_16_2, &vec_a, &vec_b, &output_16_2);

  // The clang auto-vectorization generates ordered reduce, while in zuan it
  // uses unordered reduce. The difference in the order of reduction can lead to
  // different results.
  output0.verify(output_16_2, "Dot-fp16-Zuan-16-2", 10);

  std::cerr << "Autovec = " << std::setprecision(10) << output0[0]
            << "\tzuan_16_2 = " << std::setprecision(10) << output_16_2[0]
            << std::endl;
}

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

BENCHMARK_CAPTURE(runBenchmark, zuan_16_2, _mlir_ciface_kernel_zuan_16_2)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 22);

int main(int argc, char **argv) {
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  std::cerr << "------------------------------------------------" << std::endl;
  verifyDotFp16();
  std::cerr << "------------------------------------------------" << std::endl;
  return 0;
}

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

void _mlir_ciface_kernel_transform_32_4(MemRef<float, 2> *, MemRef<float, 2> *,
                                        MemRef<float, 2> *);
void _mlir_ciface_kernel_transform_64_2(MemRef<float, 2> *, MemRef<float, 2> *,
                                        MemRef<float, 2> *);
}

using KernelFunc = void (*)(MemRef<float, 2> *, MemRef<float, 2> *,
                            MemRef<float, 2> *);

static auto initializeData(size_t m, size_t n, size_t k) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);

  MemRef<float, 2> input1({m, k}, 0);
  MemRef<float, 2> input2({n, k}, 0);

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < k; j++) {
      input1[i * k + j] = dis(gen);
    }
  }

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < k; j++) {
      input2[i * k + j] = dis(gen);
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

  autovec.verify(zuan_16_2, "matmul-zuan-16-2", 0.0001);
  autovec.verify(zuan_8_4, "matmul-zuan-8-4", 0.0001);

  MemRef<float, 2> transform_32_4({M, N}, 0);
  runKernel(_mlir_ciface_kernel_transform_32_4, &input1, &input2,
            &transform_32_4);

  MemRef<float, 2> transform_64_2({M, N}, 0);
  runKernel(_mlir_ciface_kernel_transform_64_2, &input1, &input2,
            &transform_64_2);

  autovec.verify(transform_64_2, "matmul-transform-64-2", 0.0001);
  autovec.verify(transform_32_4, "matmul-transform-32-4", 0.0001);

  // print first 10 elements
  for (int i = 0; i < 10; i++) {
    std::cerr << "Index " << i << std::setprecision(10)
              << ": autovec=" << autovec[i] << "\tzuan-8-4=" << zuan_8_4[i]
              << "\tzuan-16-2=" << zuan_16_2[i]
              << "\ttransform-32-4=" << transform_32_4[i]
              << "\ttransform-64-2=" << transform_64_2[i] << std::endl;
  }
}

//-------------------------------------------------------------------
// Zuan
//-------------------------------------------------------------------

BENCHMARK_CAPTURE(runBenchmark, zuan_8_4, _mlir_ciface_kernel_zuan_8_4)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({
        {128, 192, 256, 384, 512, 768, 1024},
        {128, 192, 256, 384, 512, 768, 1024},
        {128, 192, 256, 384, 512, 768, 1024},
    });
BENCHMARK_CAPTURE(runBenchmark, zuan_16_2, _mlir_ciface_kernel_zuan_16_2)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({
        {128, 192, 256, 384, 512, 768, 1024},
        {128, 192, 256, 384, 512, 768, 1024},
        {128, 192, 256, 384, 512, 768, 1024},
    });

//-------------------------------------------------------------------
// Transform Dialect
//-------------------------------------------------------------------

BENCHMARK_CAPTURE(runBenchmark, transform_32_4,
                  _mlir_ciface_kernel_transform_32_4)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({
        {128, 192, 256, 384, 512, 768, 1024},
        {128, 192, 256, 384, 512, 768, 1024},
        {128, 192, 256, 384, 512, 768, 1024},
    });
BENCHMARK_CAPTURE(runBenchmark, transform_64_2,
                  _mlir_ciface_kernel_transform_64_2)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({
        {128, 192, 256, 384, 512, 768, 1024},
        {128, 192, 256, 384, 512, 768, 1024},
        {128, 192, 256, 384, 512, 768, 1024},
    });

//-------------------------------------------------------------------
// Auto-vectorization
//-------------------------------------------------------------------

BENCHMARK_CAPTURE(runBenchmark, autovec_8, _mlir_ciface_kernel_autovec_8)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({
        {128, 192, 256, 384, 512, 768, 1024},
        {128, 192, 256, 384, 512, 768, 1024},
        {128, 192, 256, 384, 512, 768, 1024},
    });
BENCHMARK_CAPTURE(runBenchmark, autovec_16, _mlir_ciface_kernel_autovec_16)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({
        {128, 192, 256, 384, 512, 768, 1024},
        {128, 192, 256, 384, 512, 768, 1024},
        {128, 192, 256, 384, 512, 768, 1024},
    });
BENCHMARK_CAPTURE(runBenchmark, autovec_32, _mlir_ciface_kernel_autovec_32)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({
        {128, 192, 256, 384, 512, 768, 1024},
        {128, 192, 256, 384, 512, 768, 1024},
        {128, 192, 256, 384, 512, 768, 1024},
    });
BENCHMARK_CAPTURE(runBenchmark, autovec_64, _mlir_ciface_kernel_autovec_64)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({
        {128, 192, 256, 384, 512, 768, 1024},
        {128, 192, 256, 384, 512, 768, 1024},
        {128, 192, 256, 384, 512, 768, 1024},
    });

int main(int argc, char **argv) {
  std::cerr << "------------------------------------------------" << std::endl;
  verifyMatmul();
  std::cerr << "------------------------------------------------" << std::endl;

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}

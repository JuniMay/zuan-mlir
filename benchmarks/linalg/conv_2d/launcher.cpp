#include "benchmark/benchmark.h"
#include "common/MemRefUtils.h"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <random>

extern "C" {
void _mlir_ciface_kernel_scalar(MemRef<float, 2> *, MemRef<float, 2> *,
                                MemRef<float, 2> *);

void _mlir_ciface_kernel_autovec_8(MemRef<float, 2> *, MemRef<float, 2> *,
                                   MemRef<float, 2> *);
void _mlir_ciface_kernel_autovec_16(MemRef<float, 2> *, MemRef<float, 2> *,
                                    MemRef<float, 2> *);
void _mlir_ciface_kernel_autovec_32(MemRef<float, 2> *, MemRef<float, 2> *,
                                    MemRef<float, 2> *);
void _mlir_ciface_kernel_autovec_64(MemRef<float, 2> *, MemRef<float, 2> *,
                                    MemRef<float, 2> *);

void _mlir_ciface_kernel_dyno_8_1(MemRef<float, 2> *, MemRef<float, 2> *,
                                  MemRef<float, 2> *);
void _mlir_ciface_kernel_dyno_8_2(MemRef<float, 2> *, MemRef<float, 2> *,
                                  MemRef<float, 2> *);
void _mlir_ciface_kernel_dyno_8_4(MemRef<float, 2> *, MemRef<float, 2> *,
                                  MemRef<float, 2> *);
void _mlir_ciface_kernel_dyno_4_2(MemRef<float, 2> *, MemRef<float, 2> *,
                                  MemRef<float, 2> *);
void _mlir_ciface_kernel_dyno_4_4(MemRef<float, 2> *, MemRef<float, 2> *,
                                  MemRef<float, 2> *);
}

using KernelFunc = void (*)(MemRef<float, 2> *, MemRef<float, 2> *,
                            MemRef<float, 2> *);

static auto initializeData(size_t filter_h, size_t filter_w, size_t output_h,
                           size_t output_w) {
  auto input_h = output_h + filter_h - 1;
  auto input_w = output_w + filter_w - 1;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);

  MemRef<float, 2> input({input_h, input_w}, 0);
  MemRef<float, 2> filter({filter_h, filter_w}, 0);
  MemRef<float, 2> output({output_h, output_w}, 0);

  for (size_t i = 0; i < input_h; i++) {
    for (size_t j = 0; j < input_w; j++) {
      input[i * input_w + j] = dis(gen);
    }
  }

  for (size_t i = 0; i < filter_h; i++) {
    for (size_t j = 0; j < filter_w; j++) {
      filter[i * filter_w + j] = dis(gen);
    }
  }

  return std::make_tuple(std::move(input), std::move(filter),
                         std::move(output));
}

static void runKernel(KernelFunc kernel, MemRef<float, 2> *input,
                      MemRef<float, 2> *filter, MemRef<float, 2> *output) {
  kernel(input, filter, output);
}

static void runBenchmark(benchmark::State &state, KernelFunc kernel) {
  size_t filter_h = state.range(0);
  size_t filter_w = state.range(1);
  size_t output_h = state.range(2);
  size_t output_w = state.range(3);

  auto [input, filter, output] =
      initializeData(filter_h, filter_w, output_h, output_w);
  for (auto _ : state) {
    state.PauseTiming();
    output.fill(0);
    state.ResumeTiming();
    runKernel(kernel, &input, &filter, &output);
  }
}

static void verifyConv2d() {
  const size_t FILTER_H = 9;
  const size_t FILTER_W = 7;
  const size_t OUTPUT_H = 57;
  const size_t OUTPUT_W = 79;

  auto [input, filter, output] =
      initializeData(FILTER_H, FILTER_W, OUTPUT_H, OUTPUT_W);

  MemRef<float, 2> scalar({OUTPUT_H, OUTPUT_W}, 0);
  runKernel(_mlir_ciface_kernel_scalar, &input, &filter, &scalar);

  MemRef<float, 2> dyno_8_1({OUTPUT_H, OUTPUT_W}, 0);
  runKernel(_mlir_ciface_kernel_dyno_8_1, &input, &filter, &dyno_8_1);

  MemRef<float, 2> dyno_8_2({OUTPUT_H, OUTPUT_W}, 0);
  runKernel(_mlir_ciface_kernel_dyno_8_2, &input, &filter, &dyno_8_2);

  MemRef<float, 2> dyno_8_4({OUTPUT_H, OUTPUT_W}, 0);
  runKernel(_mlir_ciface_kernel_dyno_8_4, &input, &filter, &dyno_8_4);

  MemRef<float, 2> dyno_4_2({OUTPUT_H, OUTPUT_W}, 0);
  runKernel(_mlir_ciface_kernel_dyno_4_2, &input, &filter, &dyno_4_2);

  MemRef<float, 2> dyno_4_4({OUTPUT_H, OUTPUT_W}, 0);
  runKernel(_mlir_ciface_kernel_dyno_4_4, &input, &filter, &dyno_4_4);

  scalar.verify(dyno_8_1, "conv2d-dyno-8-1", 1e-3);
  scalar.verify(dyno_8_2, "conv2d-dyno-8-2", 1e-3);
  scalar.verify(dyno_8_4, "conv2d-dyno-8-4", 1e-3);
  scalar.verify(dyno_4_2, "conv2d-dyno-4-2", 1e-3);
  scalar.verify(dyno_4_4, "conv2d-dyno-4-4", 1e-3);

  for (size_t i = 0; i < 10; i++) {
    std::cerr << "Index " << i << std::setprecision(10)
              << ": scalar = " << scalar[i] << "\tdyno-8-1 = " << dyno_8_1[i]
              << "\tdyno-8-2 = " << dyno_8_2[i]
              << "\tdyno-8-4 = " << dyno_8_4[i]
              << "\tdyno-4-2 = " << dyno_4_2[i]
              << "\tdyno-4-4 = " << dyno_4_4[i] << std::endl;
  }
}

//-------------------------------------------------------------------
// Dyno
//-------------------------------------------------------------------

BENCHMARK_CAPTURE(runBenchmark, dyno_8_1, _mlir_ciface_kernel_dyno_8_1)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{3, 5, 7}, {3, 5, 7}, {128, 256, 512}, {128, 256, 512}});
BENCHMARK_CAPTURE(runBenchmark, dyno_8_2, _mlir_ciface_kernel_dyno_8_2)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{3, 5, 7}, {3, 5, 7}, {128, 256, 512}, {128, 256, 512}});
BENCHMARK_CAPTURE(runBenchmark, dyno_8_4, _mlir_ciface_kernel_dyno_8_4)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{3, 5, 7}, {3, 5, 7}, {128, 256, 512}, {128, 256, 512}});
BENCHMARK_CAPTURE(runBenchmark, dyno_4_2, _mlir_ciface_kernel_dyno_4_2)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{3, 5, 7}, {3, 5, 7}, {128, 256, 512}, {128, 256, 512}});
BENCHMARK_CAPTURE(runBenchmark, dyno_4_4, _mlir_ciface_kernel_dyno_4_4)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{3, 5, 7}, {3, 5, 7}, {128, 256, 512}, {128, 256, 512}});

//-------------------------------------------------------------------
// Auto-vectorization
//-------------------------------------------------------------------

BENCHMARK_CAPTURE(runBenchmark, autovec_8, _mlir_ciface_kernel_autovec_8)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{3, 5, 7}, {3, 5, 7}, {128, 256, 512}, {128, 256, 512}});
BENCHMARK_CAPTURE(runBenchmark, autovec_16, _mlir_ciface_kernel_autovec_16)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{3, 5, 7}, {3, 5, 7}, {128, 256, 512}, {128, 256, 512}});
BENCHMARK_CAPTURE(runBenchmark, autovec_32, _mlir_ciface_kernel_autovec_32)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{3, 5, 7}, {3, 5, 7}, {128, 256, 512}, {128, 256, 512}});
BENCHMARK_CAPTURE(runBenchmark, autovec_64, _mlir_ciface_kernel_autovec_64)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{3, 5, 7}, {3, 5, 7}, {128, 256, 512}, {128, 256, 512}});

int main(int argc, char **argv) {
  std::cerr << "------------------------------------------------" << std::endl;
  verifyConv2d();
  std::cerr << "------------------------------------------------" << std::endl;
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}

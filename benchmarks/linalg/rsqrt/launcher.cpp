#include "benchmark/benchmark.h"
#include "common/MemRefUtils.h"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <random>

extern "C" {
void _mlir_ciface_kernel_autovec_8(MemRef<float, 4> *, MemRef<float, 4> *);
void _mlir_ciface_kernel_autovec_16(MemRef<float, 4> *, MemRef<float, 4> *);
void _mlir_ciface_kernel_autovec_32(MemRef<float, 4> *, MemRef<float, 4> *);
void _mlir_ciface_kernel_autovec_64(MemRef<float, 4> *, MemRef<float, 4> *);

void _mlir_ciface_kernel_zuan_16_1(MemRef<float, 4> *, MemRef<float, 4> *);
void _mlir_ciface_kernel_zuan_16_2(MemRef<float, 4> *, MemRef<float, 4> *);
void _mlir_ciface_kernel_zuan_16_4(MemRef<float, 4> *, MemRef<float, 4> *);

void _mlir_ciface_kernel_zuan_16_1_est(MemRef<float, 4> *, MemRef<float, 4> *);
void _mlir_ciface_kernel_zuan_16_2_est(MemRef<float, 4> *, MemRef<float, 4> *);
void _mlir_ciface_kernel_zuan_16_4_est(MemRef<float, 4> *, MemRef<float, 4> *);

void _mlir_ciface_kernel_transform_16_1(MemRef<float, 4> *, MemRef<float, 4> *);
void _mlir_ciface_kernel_transform_16_2(MemRef<float, 4> *, MemRef<float, 4> *);
void _mlir_ciface_kernel_transform_16_4(MemRef<float, 4> *, MemRef<float, 4> *);
}

using KernelFunc = void (*)(MemRef<float, 4> *, MemRef<float, 4> *);

static auto initializeData(size_t b, size_t h, size_t w, size_t c) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);

  MemRef<float, 4> input({b, h, w, c}, 0);

  for (size_t i = 0; i < b; i++) {
    for (size_t j = 0; j < h; j++) {
      for (size_t k = 0; k < w; k++) {
        for (size_t l = 0; l < c; l++) {
          input[i * h * w * c + j * w * c + k * c + l] = dis(gen);
        }
      }
    }
  }

  return input;
}

static void runKernel(KernelFunc kernel, MemRef<float, 4> *input,
                      MemRef<float, 4> *output) {
  kernel(input, output);
}

static void runBenchmark(benchmark::State &state, KernelFunc kernel) {
  size_t b = state.range(0);
  size_t h = state.range(1);
  size_t w = state.range(2);
  size_t c = state.range(3);

  MemRef<float, 4> input = initializeData(b, h, w, c);
  MemRef<float, 4> output({b, h, w, c}, 0);
  for (auto _ : state) {
    state.PauseTiming();
    output.fill(0);
    state.ResumeTiming();
    runKernel(kernel, &input, &output);
  }
}

static void verifyRsqrt() {
  const size_t B = 123;
  const size_t H = 21;
  const size_t W = 173;
  const size_t C = 69;

  MemRef<float, 4> input = initializeData(B, H, W, C);
  MemRef<float, 4> autivec({B, H, W, C}, 0);
  runKernel(_mlir_ciface_kernel_autovec_16, &input, &autivec);

  MemRef<float, 4> zuan_16_1({B, H, W, C}, 0);
  runKernel(_mlir_ciface_kernel_zuan_16_1, &input, &zuan_16_1);

  MemRef<float, 4> zuan_16_2({B, H, W, C}, 0);
  runKernel(_mlir_ciface_kernel_zuan_16_2, &input, &zuan_16_2);

  MemRef<float, 4> zuan_16_4({B, H, W, C}, 0);
  runKernel(_mlir_ciface_kernel_zuan_16_4, &input, &zuan_16_4);

  autivec.verify(zuan_16_1, "rsqrt-zuan-16-1", 0.001);
  autivec.verify(zuan_16_1, "rsqrt-zuan-16-2", 0.001);
  autivec.verify(zuan_16_1, "rsqrt-zuan-16-4", 0.001);

  MemRef<float, 4> zuan_16_1_est({B, H, W, C}, 0);
  runKernel(_mlir_ciface_kernel_zuan_16_1_est, &input, &zuan_16_1_est);

  MemRef<float, 4> zuan_16_2_est({B, H, W, C}, 0);
  runKernel(_mlir_ciface_kernel_zuan_16_2_est, &input, &zuan_16_2_est);

  MemRef<float, 4> zuan_16_4_est({B, H, W, C}, 0);
  runKernel(_mlir_ciface_kernel_zuan_16_4_est, &input, &zuan_16_4_est);

  // Zuan Compiler uses vfrsqrt7 intrinsic, while clang uses accurate rsqrt
  // even if fast-math is enabled.
  autivec.verify(zuan_16_1_est, "rsqrt-zuan-16-1-est",
                 100); // TODO: verify 7-bit estimate
  autivec.verify(zuan_16_2_est, "rsqrt-zuan-16-2-est",
                 100); // TODO: verify 7-bit estimate
  autivec.verify(zuan_16_4_est, "rsqrt-zuan-16-4-est",
                 100); // TODO: verify 7-bit estimate

  MemRef<float, 4> transform_16_1({B, H, W, C}, 0);
  runKernel(_mlir_ciface_kernel_transform_16_1, &input, &transform_16_1);

  MemRef<float, 4> transform_16_2({B, H, W, C}, 0);
  runKernel(_mlir_ciface_kernel_transform_16_2, &input, &transform_16_2);

  MemRef<float, 4> transform_16_4({B, H, W, C}, 0);
  runKernel(_mlir_ciface_kernel_transform_16_4, &input, &transform_16_4);

  autivec.verify(transform_16_1, "rsqrt-transform-16-1", 0.001);
  autivec.verify(transform_16_1, "rsqrt-transform-16-2", 0.001);
  autivec.verify(transform_16_1, "rsqrt-transform-16-4", 0.001);

  for (size_t i = 0; i < 10; i++) {
    std::cerr << "Index " << i << std::setprecision(10)
              << ": autovec = " << autivec[i]
              << "\tzuan-16-1 = " << zuan_16_1[i]
              << "\tzuan-16-2 = " << zuan_16_2[i]
              << "\tzuan-16-4 = " << zuan_16_4[i]
              << "\tzuan-16-1-est = " << zuan_16_1_est[i]
              << "\tzuan-16-2-est = " << zuan_16_2_est[i]
              << "\tzuan-16-4-est = " << zuan_16_4_est[i]
              << "\ttransform-16-1 = " << transform_16_1[i]
              << "\ttransform-16-2 = " << transform_16_2[i]
              << "\ttransform-16-4 = " << transform_16_4[i] << std::endl;
  }
}

//-------------------------------------------------------------------
// Zuan
//-------------------------------------------------------------------
BENCHMARK_CAPTURE(runBenchmark, zuan_16_1, _mlir_ciface_kernel_zuan_16_1)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{32, 64, 128}, {32, 64, 128}, {32, 64, 128}, {32, 64, 128}});
BENCHMARK_CAPTURE(runBenchmark, zuan_16_2, _mlir_ciface_kernel_zuan_16_2)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{32, 64, 128}, {32, 64, 128}, {32, 64, 128}, {32, 64, 128}});
BENCHMARK_CAPTURE(runBenchmark, zuan_16_4, _mlir_ciface_kernel_zuan_16_4)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{32, 64, 128}, {32, 64, 128}, {32, 64, 128}, {32, 64, 128}});

BENCHMARK_CAPTURE(runBenchmark, zuan_16_1_est,
                  _mlir_ciface_kernel_zuan_16_1_est)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{32, 64, 128}, {32, 64, 128}, {32, 64, 128}, {32, 64, 128}});
BENCHMARK_CAPTURE(runBenchmark, zuan_16_2_est,
                  _mlir_ciface_kernel_zuan_16_2_est)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{32, 64, 128}, {32, 64, 128}, {32, 64, 128}, {32, 64, 128}});
BENCHMARK_CAPTURE(runBenchmark, zuan_16_4_est,
                  _mlir_ciface_kernel_zuan_16_4_est)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{32, 64, 128}, {32, 64, 128}, {32, 64, 128}, {32, 64, 128}});

//-------------------------------------------------------------------
// Transform Dialect
//-------------------------------------------------------------------

BENCHMARK_CAPTURE(runBenchmark, transform_16_1,
                  _mlir_ciface_kernel_transform_16_1)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{32, 64, 128}, {32, 64, 128}, {32, 64, 128}, {32, 64, 128}});
BENCHMARK_CAPTURE(runBenchmark, transform_16_2,
                  _mlir_ciface_kernel_transform_16_2)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{32, 64, 128}, {32, 64, 128}, {32, 64, 128}, {32, 64, 128}});
BENCHMARK_CAPTURE(runBenchmark, transform_16_4,
                  _mlir_ciface_kernel_transform_16_4)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{32, 64, 128}, {32, 64, 128}, {32, 64, 128}, {32, 64, 128}});

//-------------------------------------------------------------------
// Auto-vectorization
//-------------------------------------------------------------------

BENCHMARK_CAPTURE(runBenchmark, autovec_8, _mlir_ciface_kernel_autovec_8)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{32, 64, 128}, {32, 64, 128}, {32, 64, 128}, {32, 64, 128}});
BENCHMARK_CAPTURE(runBenchmark, autovec_16, _mlir_ciface_kernel_autovec_16)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{32, 64, 128}, {32, 64, 128}, {32, 64, 128}, {32, 64, 128}});
BENCHMARK_CAPTURE(runBenchmark, autovec_32, _mlir_ciface_kernel_autovec_32)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{32, 64, 128}, {32, 64, 128}, {32, 64, 128}, {32, 64, 128}});
BENCHMARK_CAPTURE(runBenchmark, autovec_64, _mlir_ciface_kernel_autovec_64)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{32, 64, 128}, {32, 64, 128}, {32, 64, 128}, {32, 64, 128}});

int main(int argc, char **argv) {
  std::cerr << "------------------------------------------------" << std::endl;
  verifyRsqrt();
  std::cerr << "------------------------------------------------" << std::endl;
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}

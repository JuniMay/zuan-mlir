#include "benchmark/benchmark.h"
#include "common/verification.h"
#include "triton/matmul/matmul.h"
#include <cmath>
#include <random>
#include <vector>

const uint32_t BLOCK_SIZE_M = 64;
const uint32_t BLOCK_SIZE_N = 64;

static auto initializeData(size_t m, size_t n, size_t k) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);

  std::vector<float> input1(m * k);
  std::vector<float> input2(k * n);
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < k; j++) {
      input1[i * k + j] = dis(gen);
    }
  }

  for (size_t i = 0; i < k; i++) {
    for (size_t j = 0; j < n; j++) {
      input2[i * n + j] = dis(gen);
    }
  }

  return std::make_tuple(std::move(input1), std::move(input2));
}

static void runKernel(kernel_ptr_t kernel, uint32_t m, uint32_t n, uint32_t k,
                      float *input1, float *input2, float *output) {
  uint32_t gridX = std::ceil(n / static_cast<float>(BLOCK_SIZE_N)) *
                   std::ceil(m / static_cast<float>(BLOCK_SIZE_M));
  launch_kernel(gridX, 1, 1, kernel, input1, input2, output, m, n, k, k, 1, n,
                1, n, 1);
}

static void runBenchmark(benchmark::State &state, kernel_ptr_t kernel) {
  size_t m = state.range(0);
  size_t n = state.range(1);
  size_t k = state.range(2);

  auto [input1, input2] = initializeData(m, n, k);
  std::vector<float> output(m * n);
  for (auto _ : state) {
    runKernel(kernel, m, n, k, input1.data(), input2.data(), output.data());
  }
}

BENCHMARK_CAPTURE(runBenchmark, triton_cpu, kernel_triton_cpu)
    ->Unit(benchmark::kMillisecond)
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Args({511, 237, 123})
    ->Args({1023, 509, 2173});

BENCHMARK_CAPTURE(runBenchmark, zuan, kernel_zuan_wrapper)
    ->Unit(benchmark::kMillisecond)
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Args({511, 237, 123})
    ->Args({1023, 509, 2173});

void verify() {
  const size_t M = 511;
  const size_t N = 237;
  const size_t K = 123;

  auto [input1, input2] = initializeData(M, N, K);
  std::vector<float> output_triton_cpu(M * N);
  std::vector<float> output_zuan(M * N);
  runKernel(kernel_triton_cpu, M, N, K, input1.data(), input2.data(),
            output_triton_cpu.data());
  runKernel(kernel_zuan_wrapper, M, N, K, input1.data(), input2.data(),
            output_zuan.data());

  verify<float>(output_triton_cpu.data(), output_zuan.data(), M * N, "Matmul",
                0.0001);
}

int main(int argc, char **argv) {
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();

  verify();

  return 0;
}
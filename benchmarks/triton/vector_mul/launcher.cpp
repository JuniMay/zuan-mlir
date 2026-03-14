#include "benchmark/benchmark.h"
#include "common/verification.h"
#include "triton/vector_mul/vector_mul.h"
#include <cmath>
#include <random>
#include <vector>

const uint32_t BLOCK_SIZE = 1024;

static auto initializeData(size_t n_elements) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  std::vector<float> vec_a(n_elements);
  std::vector<float> vec_b(n_elements);
  for (size_t i = 0; i < n_elements; i++) {
    vec_a[i] = dis(gen);
    vec_b[i] = dis(gen);
  }

  return std::make_tuple(std::move(vec_a), std::move(vec_b));
}

static void runKernel(kernel_ptr_t kernel, uint32_t n_elements, float *vec_a,
                      float *vec_b, float *vec_c) {
  uint32_t gridX = std::ceil(n_elements / static_cast<float>(BLOCK_SIZE));
  launch_kernel(gridX, 1, 1, kernel, vec_a, vec_b, vec_c, n_elements);
}

static void runBenchmark(benchmark::State &state, kernel_ptr_t kernel) {
  size_t n_elements = state.range(0);
  auto [vec_a, vec_b] = initializeData(n_elements);
  std::vector<float> vec_c(n_elements);
  for (auto _ : state) {
    runKernel(kernel, n_elements, vec_a.data(), vec_b.data(), vec_c.data());
  }
}

BENCHMARK_CAPTURE(runBenchmark, triton_cpu, kernel_triton_cpu)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1 << 13)
    ->Arg(1 << 14)
    ->Arg(1 << 15)
    ->Arg(1 << 16);

BENCHMARK_CAPTURE(runBenchmark, dyno, kernel_dyno_wrapper)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1 << 13)
    ->Arg(1 << 14)
    ->Arg(1 << 15)
    ->Arg(1 << 16);

void verify() {
  const size_t N = 12937;

  auto [vec_a, vec_b] = initializeData(N);
  std::vector<float> vec_c_scalar(N);
  std::vector<float> vec_c_triton_cpu(N);
  std::vector<float> vec_c_dyno(N);
  runKernel(kernel_scalar_wrapper, N, vec_a.data(), vec_b.data(),
            vec_c_scalar.data());
  runKernel(kernel_triton_cpu, N, vec_a.data(), vec_b.data(),
            vec_c_triton_cpu.data());
  runKernel(kernel_dyno_wrapper, N, vec_a.data(), vec_b.data(),
            vec_c_dyno.data());

  verify(vec_c_scalar.data(), vec_c_triton_cpu.data(), N,
         "Vector Multiply Triton CPU");
  verify(vec_c_scalar.data(), vec_c_dyno.data(), N, "Vector Multiply Dyno");
}

int main(int argc, char **argv) {
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();

  verify();

  return 0;
}

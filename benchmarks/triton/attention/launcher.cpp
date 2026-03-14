#include "benchmark/benchmark.h"
#include "common/verification.h"
#include "triton/attention/attention.h"
#include <cmath>
#include <random>
#include <tuple>
#include <vector>

const uint32_t HEAD_DIM = 64;
const float SM_SCALE = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

static auto initializeData(size_t seq_len) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  std::vector<float> q(seq_len * HEAD_DIM);
  std::vector<float> k(seq_len * HEAD_DIM);
  for (size_t i = 0; i < seq_len * HEAD_DIM; ++i) {
    q[i] = dis(gen);
    k[i] = dis(gen);
  }

  return std::make_tuple(std::move(q), std::move(k));
}

static void runKernel(kernel_ptr_t kernel, uint32_t seq_len, float *q, float *k,
                      float *output) {
  uint32_t gridX =
      std::ceil(seq_len / 64.0f) * std::ceil(seq_len / 128.0f);
  launch_kernel(gridX, 1, 1, kernel, q, k, output, seq_len, seq_len, HEAD_DIM,
                HEAD_DIM, 1, 1, HEAD_DIM, seq_len, 1, SM_SCALE);
}

static void runBenchmark(benchmark::State &state, kernel_ptr_t kernel) {
  size_t seq_len = state.range(0);
  auto [q, k] = initializeData(seq_len);
  std::vector<float> output(seq_len * seq_len);
  for (auto _ : state) {
    runKernel(kernel, seq_len, q.data(), k.data(), output.data());
  }
}

BENCHMARK_CAPTURE(runBenchmark, triton_cpu, kernel_triton_cpu)
    ->Unit(benchmark::kMillisecond)
    ->Arg(64)
    ->Arg(128)
    ->Arg(192)
    ->Arg(256);

BENCHMARK_CAPTURE(runBenchmark, zuan, kernel_zuan_wrapper)
    ->Unit(benchmark::kMillisecond)
    ->Arg(64)
    ->Arg(128)
    ->Arg(192)
    ->Arg(256);

void verify() {
  const size_t SEQ_LEN = 113;

  auto [q, k] = initializeData(SEQ_LEN);
  std::vector<float> output_scalar(SEQ_LEN * SEQ_LEN);
  std::vector<float> output_triton_cpu(SEQ_LEN * SEQ_LEN);
  std::vector<float> output_zuan(SEQ_LEN * SEQ_LEN);
  runKernel(kernel_scalar_wrapper, SEQ_LEN, q.data(), k.data(),
            output_scalar.data());
  runKernel(kernel_triton_cpu, SEQ_LEN, q.data(), k.data(),
            output_triton_cpu.data());
  runKernel(kernel_zuan_wrapper, SEQ_LEN, q.data(), k.data(),
            output_zuan.data());

  verify<float>(output_scalar.data(), output_triton_cpu.data(),
                SEQ_LEN * SEQ_LEN, "Attention Triton CPU", 0.001);
  verify<float>(output_scalar.data(), output_zuan.data(), SEQ_LEN * SEQ_LEN,
                "Attention Zuan", 0.001);
}

int main(int argc, char **argv) {
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();

  verify();

  return 0;
}

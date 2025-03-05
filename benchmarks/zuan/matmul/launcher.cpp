#include "benchmark/benchmark.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>

extern "C" {
void _mlir_ciface_matmul(StridedMemRefType<float, 2> *,
                         StridedMemRefType<float, 2> *,
                         StridedMemRefType<float, 2> *);
}

// Simple naive implementation with loop interchange.
void naive_matmul(StridedMemRefType<float, 2> *a,
                  StridedMemRefType<float, 2> *b,
                  StridedMemRefType<float, 2> *c) {
  int64_t M = a->sizes[0];
  int64_t N = b->sizes[1];
  int64_t K = a->sizes[1];

  for (int64_t i = 0; i < M; i++) {
    for (int64_t k = 0; k < K; k++) {
      for (int64_t j = 0; j < N; j++) {
        (*c)[i][j] += (*a)[i][k] * (*b)[k][j];
      }
    }
  }
}

template <typename T, int N>
StridedMemRefType<T, N> createMemRef(const std::vector<size_t> sizes, T init) {
  StridedMemRefType<T, N> memref;
  assert(sizes.size() == N && "dimension mismatch");
  std::copy(sizes.begin(), sizes.end(), memref.sizes);
  memref.offset = 0;
  if (N > 0) {
    memref.strides[N - 1] = 1;
    for (int i = N - 1; i > 0; i--) {
      memref.strides[i - 1] = memref.strides[i] * memref.sizes[i];
    }
  }
  size_t totalSize =
      std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<size_t>());
  memref.basePtr = static_cast<T *>(malloc(totalSize * sizeof(T)));
  std::fill(memref.basePtr, memref.basePtr + totalSize, init);
  memref.data = memref.basePtr;
  return memref;
}


template <typename T, int N> void fillMemRef(StridedMemRefType<T, N> memref, T val) {
  size_t totalSize =
      std::accumulate(memref.sizes, memref.sizes + N, 1, std::multiplies<size_t>());
  std::fill(memref.data, memref.data + totalSize, val);
}

template <typename T>
void verify(T *A, T *B, int size, const std::string &name) {
  const std::string PASS = "\033[32mPASS\033[0m";
  const std::string FAIL = "\033[31mFAIL\033[0m";

  std::cout << name << " ";
  if (!A || !B) {
    std::cout << FAIL << " (Null pointer detected)" << std::endl;
    return;
  }

  bool isPass = true;
  for (int i = 0; i < size; ++i) {
    if (fabs(A[i] - B[i]) > 0.0001) {
      std::cout << FAIL << std::endl;
      std::cout << "Index " << i << ":\tA=" << std::setprecision(10) << A[i]
                << " B=" << std::setprecision(10) << B[i] << std::endl;
      isPass = false;
      break;
    }
  }
  if (isPass) {
    std::cout << PASS << std::endl;
  }
}

template <typename T, int N> void freeMemRef(StridedMemRefType<T, N> memref) {
  free(memref.basePtr);
}

using KernelFunc = void (*)(StridedMemRefType<float, 2> *,
                            StridedMemRefType<float, 2> *,
                            StridedMemRefType<float, 2> *);

static auto initializeData(size_t m, size_t n, size_t k) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);

  auto input1 = createMemRef<float, 2>({m, k}, 0);
  auto input2 = createMemRef<float, 2>({k, n}, 0);
  auto output = createMemRef<float, 2>({m, n}, 0);

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < k; j++) {
      input1[i][j] = dis(gen);
    }
  }

  for (size_t i = 0; i < k; i++) {
    for (size_t j = 0; j < n; j++) {
      input2[i][j] = dis(gen);
    }
  }

  return std::make_tuple(input1, input2, output);
}

static void runKernel(KernelFunc kernel, StridedMemRefType<float, 2> *input1,
                      StridedMemRefType<float, 2> *input2,
                      StridedMemRefType<float, 2> *output) {
  kernel(input1, input2, output);
}

static void runBenchmark(benchmark::State &state, KernelFunc kernel) {
  size_t m = state.range(0);
  size_t n = state.range(1);
  size_t k = state.range(2);

  auto [input1, input2, output] = initializeData(m, n, k);
  for (auto _ : state) {
    state.PauseTiming();
    fillMemRef<float>(output, 0);
    state.ResumeTiming();
    runKernel(kernel, &input1, &input2, &output);
  }
}

static void verifyMatmul() {
  const size_t M = 511;
  const size_t N = 237;
  const size_t K = 123;

  auto [input1, input2, naive_buffer] = initializeData(M, N, K);
  auto zuan_buffer = createMemRef<float, 2>({M, N}, 0);

  runKernel(naive_matmul, &input1, &input2, &naive_buffer);
  runKernel(_mlir_ciface_matmul, &input1, &input2, &zuan_buffer);

  verify(naive_buffer.data, zuan_buffer.data, M * N, "Matmul");

  // print first 10 elements
  for (int i = 0; i < 10; i++) {
    std::cout << "Index " << i << ":\tNaive=" << std::setprecision(10)
              << naive_buffer.data[i] << " Zuan=" << std::setprecision(10)
              << zuan_buffer.data[i] << std::endl;
  }

  freeMemRef(input1);
  freeMemRef(input2);
  freeMemRef(naive_buffer);
  freeMemRef(zuan_buffer);
}

BENCHMARK_CAPTURE(runBenchmark, zuan, _mlir_ciface_matmul)
    ->Unit(benchmark::kMillisecond)
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024});

BENCHMARK_CAPTURE(runBenchmark, naive, naive_matmul)
    ->Unit(benchmark::kMillisecond)
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024});

int main(int argc, char **argv) {
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  std::cout << "------------------------------------------------" << std::endl;
  verifyMatmul();
  std::cout << "------------------------------------------------" << std::endl;

  return 0;
}

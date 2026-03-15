#include "RegressionUtils.h"

#include <cstddef>
#include <cstdint>

extern "C" {
void _mlir_ciface_kernel_dyno(MemRef<int32_t, 3> *, MemRef<int32_t, 1> *);
}

namespace {

constexpr int64_t kDim0 = 37;
constexpr int64_t kDim1 = 7;
constexpr int64_t kDim2 = 11;

int32_t makeSource(int64_t i, int64_t j, int64_t k) {
  return static_cast<int32_t>(((i * 7 + j * 3 + k * 5) % 19) - 9);
}

void initializeInput(MemRef<int32_t, 3> &source) {
  for (int64_t i = 0; i < kDim0; ++i) {
    for (int64_t j = 0; j < kDim1; ++j) {
      for (int64_t k = 0; k < kDim2; ++k) {
        source[(i * kDim1 + j) * kDim2 + k] = makeSource(i, j, k);
      }
    }
  }
}

int32_t runReference(const MemRef<int32_t, 3> &source) {
  int32_t acc = 0;
  for (int64_t i = 0; i < kDim0; ++i) {
    for (int64_t j = 0; j < kDim1; ++j) {
      for (int64_t k = 0; k < kDim2; ++k) {
        acc += source[(i * kDim1 + j) * kDim2 + k];
      }
    }
  }
  return acc;
}

} // namespace

int main() {
  dyno::regression::logReferenceMode("factorized-integer-strict");

  MemRef<int32_t, 3> source({kDim0, kDim1, kDim2}, 0);
  MemRef<int32_t, 1> expected({1}, 0);
  MemRef<int32_t, 1> actual({1}, 0);

  initializeInput(source);
  expected[0] = runReference(source);
  _mlir_ciface_kernel_dyno(&source, &actual);

  dyno::regression::verifyMemRef(actual, expected, DYNO_REGRESSION_NAME);
  return 0;
}

#include "RegressionUtils.h"

#include <cstddef>

extern "C" {
void _mlir_ciface_kernel_dyno(MemRef<float, 3> *, MemRef<float, 1> *);
}

namespace {

constexpr int64_t kDim0 = 5;
constexpr int64_t kDim1 = 7;
constexpr int64_t kDim2 = 11;
constexpr float kEpsilon = static_cast<float>(DYNO_REGRESSION_EPSILON);

float makeSource(int64_t i, int64_t j, int64_t k) {
  int64_t linear = (i * kDim1 + j) * kDim2 + k;
  switch (linear % 6) {
  case 0:
    return 100000000.0f;
  case 1:
    return static_cast<float>((j % 5) + 1) * 0.25f;
  case 2:
    return -100000000.0f;
  case 3:
    return 1.0f;
  case 4:
    return -0.5f;
  default:
    return static_cast<float>((k % 7) + 1) * 0.125f;
  }
}

void initializeInput(MemRef<float, 3> &source) {
  for (int64_t i = 0; i < kDim0; ++i) {
    for (int64_t j = 0; j < kDim1; ++j) {
      for (int64_t k = 0; k < kDim2; ++k) {
        source[(i * kDim1 + j) * kDim2 + k] = makeSource(i, j, k);
      }
    }
  }
}

float runReference(const MemRef<float, 3> &source) {
  float acc = 0.0f;
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
  dyno::regression::logReferenceMode("ordered-strict");

  MemRef<float, 3> source({kDim0, kDim1, kDim2}, 0.0f);
  MemRef<float, 1> expected({1}, 0.0f);
  MemRef<float, 1> actual({1}, 0.0f);

  initializeInput(source);
  expected[0] = runReference(source);
  _mlir_ciface_kernel_dyno(&source, &actual);

  dyno::regression::verifyMemRef(actual, expected, DYNO_REGRESSION_NAME,
                                 kEpsilon);
  return 0;
}

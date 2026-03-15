#include "RegressionUtils.h"

#include <cstddef>

extern "C" {
void _mlir_ciface_kernel_dyno(MemRef<float, 2> *, MemRef<float, 1> *);
}

namespace {

constexpr int64_t kRows = 7;
constexpr int64_t kCols = 11;
constexpr float kEpsilon = static_cast<float>(DYNO_REGRESSION_EPSILON);

float makeSource(int64_t row, int64_t col) {
  switch (row) {
  case 0:
    return 100000000.0f;
  case 1:
    return static_cast<float>((col % 5) + 1) * 0.25f;
  case 2:
    return -100000000.0f;
  case 3:
    return 1.0f;
  case 4:
    return -0.5f;
  case 5:
    return 2.0f;
  default:
    return static_cast<float>(col + 1) * 0.125f;
  }
}

float makeInit(int64_t col) {
  return static_cast<float>(col % 3) * 0.5f;
}

void initializeInputs(MemRef<float, 2> &source, MemRef<float, 1> &dst) {
  for (int64_t row = 0; row < kRows; ++row) {
    for (int64_t col = 0; col < kCols; ++col) {
      source[row * kCols + col] = makeSource(row, col);
    }
  }

  for (int64_t col = 0; col < kCols; ++col) {
    dst[col] = makeInit(col);
  }
}

void runReference(const MemRef<float, 2> &source, MemRef<float, 1> &dst) {
  for (int64_t col = 0; col < kCols; ++col) {
    float acc = dst[col];
    for (int64_t row = 0; row < kRows; ++row) {
      acc += source[row * kCols + col];
    }
    dst[col] = acc;
  }
}

} // namespace

int main() {
  dyno::regression::logReferenceMode("ordered-strict");

  MemRef<float, 2> source({kRows, kCols}, 0.0f);
  MemRef<float, 1> expected({kCols}, 0.0f);
  MemRef<float, 1> actual({kCols}, 0.0f);

  initializeInputs(source, expected);
  initializeInputs(source, actual);
  runReference(source, expected);
  _mlir_ciface_kernel_dyno(&source, &actual);

  dyno::regression::verifyMemRef(actual, expected, DYNO_REGRESSION_NAME,
                                 kEpsilon);
  return 0;
}

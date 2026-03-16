#include "RegressionUtils.h"

#include <cstddef>

extern "C" {
void _mlir_ciface_kernel_dyno(MemRef<int32_t, 2> *, MemRef<int32_t, 1> *);
}

namespace {

constexpr int64_t kRows = 7;
constexpr int64_t kCols = 11;

int32_t makeSource(int64_t row, int64_t col) {
  return static_cast<int32_t>(((row * 7 + col * 3) % 13) - 6);
}

int32_t makeInit(int64_t col) { return static_cast<int32_t>(col - 5); }

void initializeInputs(MemRef<int32_t, 2> &source, MemRef<int32_t, 1> &dst) {
  for (int64_t row = 0; row < kRows; ++row) {
    for (int64_t col = 0; col < kCols; ++col) {
      source[row * kCols + col] = makeSource(row, col);
    }
  }

  for (int64_t col = 0; col < kCols; ++col) {
    dst[col] = makeInit(col);
  }
}

void runReference(const MemRef<int32_t, 2> &source, MemRef<int32_t, 1> &dst) {
  for (int64_t col = 0; col < kCols; ++col) {
    int32_t acc = dst[col];
    for (int64_t row = 0; row < kRows; ++row) {
      acc += source[row * kCols + col];
    }
    dst[col] = acc;
  }
}

} // namespace

int main() {
  MemRef<int32_t, 2> source({kRows, kCols}, 0);
  MemRef<int32_t, 1> expected({kCols}, 0);
  MemRef<int32_t, 1> actual({kCols}, 0);

  initializeInputs(source, expected);
  initializeInputs(source, actual);
  runReference(source, expected);
  _mlir_ciface_kernel_dyno(&source, &actual);

  dyno::regression::verifyMemRef(actual, expected, DYNO_REGRESSION_NAME);
  return 0;
}

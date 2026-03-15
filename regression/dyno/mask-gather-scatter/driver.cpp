#include "RegressionUtils.h"

#include <cstddef>
#include <cstdint>

extern "C" {
void _mlir_ciface_kernel_dyno(MemRef<float, 2> *, MemRef<int64_t, 1> *,
                              MemRef<int64_t, 1> *, MemRef<bool, 1> *,
                              MemRef<float, 1> *);
}

namespace {

constexpr int64_t kRows = 7;
constexpr int64_t kCols = 9;
constexpr int64_t kLen = 13;
constexpr float kSentinel = -999.0f;

void initializeBase(MemRef<float, 2> &source) {
  for (int64_t row = 0; row < kRows; ++row) {
    for (int64_t col = 0; col < kCols; ++col) {
      source[row * kCols + col] =
          static_cast<float>(row * 100 + col * 7 - 20);
    }
  }
}

void initializeIndices(MemRef<int64_t, 1> &idx0, MemRef<int64_t, 1> &idx1,
                       MemRef<bool, 1> &mask) {
  const int64_t idx0Values[kLen] = {3, 1, 6, 1, 0, 5, 2, 6, 4, 2, 3, 0, 5};
  const int64_t idx1Values[kLen] = {8, 0, 4, 0, 7, 1, 5, 4, 2, 5, 8, 6, 1};
  const bool maskValues[kLen] = {true, true, false, true, true,  false, true,
                                 true, true, false, true, true,  true};

  for (int64_t i = 0; i < kLen; ++i) {
    idx0[i] = idx0Values[i];
    idx1[i] = idx1Values[i];
    mask[i] = maskValues[i];
  }
}

void runReference(const MemRef<float, 2> &source, const MemRef<int64_t, 1> &idx0,
                  const MemRef<int64_t, 1> &idx1, const MemRef<bool, 1> &mask,
                  MemRef<float, 1> &dst) {
  for (int64_t i = 0; i < kLen; ++i) {
    if (!mask[i]) {
      continue;
    }
    dst[idx0[i]] = source[idx0[i] * kCols + idx1[i]];
  }
}

} // namespace

int main() {
  MemRef<float, 2> source({kRows, kCols}, 0.0f);
  MemRef<int64_t, 1> idx0({kLen}, 0);
  MemRef<int64_t, 1> idx1({kLen}, 0);
  MemRef<bool, 1> mask({kLen}, false);
  MemRef<float, 1> expected({kRows}, kSentinel);
  MemRef<float, 1> actual({kRows}, kSentinel);

  initializeBase(source);
  initializeIndices(idx0, idx1, mask);
  runReference(source, idx0, idx1, mask, expected);
  _mlir_ciface_kernel_dyno(&source, &idx0, &idx1, &mask, &actual);

  dyno::regression::verifyMemRef(actual, expected, DYNO_REGRESSION_NAME);
  return 0;
}

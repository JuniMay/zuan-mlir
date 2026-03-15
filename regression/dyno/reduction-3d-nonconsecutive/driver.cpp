#include "RegressionUtils.h"

#include <cstddef>
#include <cstdint>

extern "C" {
void _mlir_ciface_kernel_dyno(MemRef<int32_t, 3> *, MemRef<int32_t, 1> *,
                              MemRef<int32_t, 1> *);
}

namespace {

constexpr int64_t kDim0 = 5;
constexpr int64_t kDim1 = 23;
constexpr int64_t kDim2 = 7;

int32_t makeSource(int64_t i, int64_t j, int64_t k) {
  return static_cast<int32_t>(((i * 11 + j * 5 + k * 3) % 17) - 8);
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

void runReference(const MemRef<int32_t, 3> &source, MemRef<int32_t, 1> &partial,
                  MemRef<int32_t, 1> &final) {
  int32_t total = 0;
  for (int64_t j = 0; j < kDim1; ++j) {
    int32_t acc = 0;
    for (int64_t i = 0; i < kDim0; ++i) {
      for (int64_t k = 0; k < kDim2; ++k) {
        acc += source[(i * kDim1 + j) * kDim2 + k];
      }
    }
    partial[j] = acc;
    total += acc;
  }
  final[0] = total;
}

} // namespace

int main() {
  dyno::regression::logReferenceMode("factorized-integer-strict");

  MemRef<int32_t, 3> source({kDim0, kDim1, kDim2}, 0);
  MemRef<int32_t, 1> expectedPartial({kDim1}, 0);
  MemRef<int32_t, 1> actualPartial({kDim1}, 0);
  MemRef<int32_t, 1> expectedFinal({1}, 0);
  MemRef<int32_t, 1> actualFinal({1}, 0);

  initializeInput(source);
  runReference(source, expectedPartial, expectedFinal);
  _mlir_ciface_kernel_dyno(&source, &actualPartial, &actualFinal);

  dyno::regression::verifyMemRef(actualPartial, expectedPartial,
                                 DYNO_REGRESSION_NAME "-partial");
  dyno::regression::verifyMemRef(actualFinal, expectedFinal,
                                 DYNO_REGRESSION_NAME "-final");
  return 0;
}

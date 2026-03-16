#include "RegressionUtils.h"

#include <cstddef>

extern "C" {
void _mlir_ciface_kernel_dyno(MemRef<float, 2> *, MemRef<float, 2> *,
                              MemRef<float, 2> *);
}

namespace {

constexpr int64_t kM = 11;
constexpr int64_t kN = 13;
constexpr int64_t kK = 17;
constexpr float kEpsilon = static_cast<float>(DYNO_REGRESSION_EPSILON);

float makeLhs(int64_t k, int64_t m) {
  return static_cast<float>(((k * 11 + m * 7) % 29) - 14) * 0.125f;
}

float makeRhs(int64_t k, int64_t n) {
  return static_cast<float>(((k * 5 + n * 3) % 17) - 8) * 0.25f;
}

void initializeInputs(MemRef<float, 2> &lhs, MemRef<float, 2> &rhs) {
  for (int64_t k = 0; k < kK; ++k) {
    for (int64_t m = 0; m < kM; ++m) {
      lhs[k * kM + m] = makeLhs(k, m);
    }
  }

  for (int64_t k = 0; k < kK; ++k) {
    for (int64_t n = 0; n < kN; ++n) {
      rhs[k * kN + n] = makeRhs(k, n);
    }
  }
}

void runReference(const MemRef<float, 2> &lhs, const MemRef<float, 2> &rhs,
                  MemRef<float, 2> &dst) {
  for (int64_t m = 0; m < kM; ++m) {
    for (int64_t n = 0; n < kN; ++n) {
      float acc = dst[m * kN + n];
      for (int64_t k = 0; k < kK; ++k) {
        acc += lhs[k * kM + m] * rhs[k * kN + n];
      }
      dst[m * kN + n] = acc;
    }
  }
}

} // namespace

int main() {
  dyno::regression::logReferenceMode("ordered-strict");

  MemRef<float, 2> lhs({kK, kM}, 0.0f);
  MemRef<float, 2> rhs({kK, kN}, 0.0f);
  MemRef<float, 2> expected({kM, kN}, 0.0f);
  MemRef<float, 2> actual({kM, kN}, 0.0f);

  initializeInputs(lhs, rhs);
  runReference(lhs, rhs, expected);
  _mlir_ciface_kernel_dyno(&lhs, &rhs, &actual);

  dyno::regression::verifyMemRef(actual, expected, DYNO_REGRESSION_NAME,
                                 kEpsilon);
  return 0;
}

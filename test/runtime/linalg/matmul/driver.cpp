#include "RegressionUtils.h"

#include <cstddef>

extern "C" {
void _mlir_ciface_kernel_dyno(MemRef<float, 2> *, MemRef<float, 2> *,
                              MemRef<float, 2> *);
}

namespace {

constexpr int64_t kM = 17;
constexpr int64_t kN = 19;
constexpr int64_t kK = 13;
constexpr float kEpsilon = static_cast<float>(DYNO_REGRESSION_EPSILON);

float makeLhs(int64_t i, int64_t k) {
  return static_cast<float>(((i * 7 + k * 5) % 23) - 11) * 0.25f;
}

float makeRhs(int64_t k, int64_t j) {
  return static_cast<float>(((k * 3 + j * 9) % 19) - 9) * 0.2f;
}

void initializeInputs(MemRef<float, 2> &lhs, MemRef<float, 2> &rhs) {
  for (int64_t i = 0; i < kM; ++i) {
    for (int64_t k = 0; k < kK; ++k) {
      lhs[i * kK + k] = makeLhs(i, k);
    }
  }

  for (int64_t k = 0; k < kK; ++k) {
    for (int64_t j = 0; j < kN; ++j) {
      rhs[k * kN + j] = makeRhs(k, j);
    }
  }
}

void runReference(const MemRef<float, 2> &lhs, const MemRef<float, 2> &rhs,
                  MemRef<float, 2> &dst) {
  for (int64_t i = 0; i < kM; ++i) {
    for (int64_t j = 0; j < kN; ++j) {
      float acc = dst[i * kN + j];
      for (int64_t k = 0; k < kK; ++k) {
        acc += lhs[i * kK + k] * rhs[k * kN + j];
      }
      dst[i * kN + j] = acc;
    }
  }
}

} // namespace

int main() {
  dyno::regression::logReferenceMode("ordered-strict");

  MemRef<float, 2> lhs({kM, kK}, 0.0f);
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

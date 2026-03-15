#include "RegressionUtils.h"

#include <cstddef>

extern "C" {
void _mlir_ciface_kernel_dyno(MemRef<float, 1> *, MemRef<float, 0> *);
}

namespace {

constexpr int64_t kN = 139;
constexpr float kEpsilon = static_cast<float>(DYNO_REGRESSION_EPSILON);

float makeElement(int64_t index) {
  // Keep the lane-grouped relaxed reference exact across targets: a
  // cancellation-heavy pattern would make the result depend on the backend's
  // final lane-fold order, which this regression does not intend to test.
  return static_cast<float>((index % 7) - 3);
}

void initializeInput(MemRef<float, 1> &source) {
  for (int64_t i = 0; i < kN; ++i) {
    source[i] = makeElement(i);
  }
}

} // namespace

int main() {
  dyno::regression::logReferenceMode("parallel-lane-grouped-relaxed");

  MemRef<float, 1> source({kN}, 0.0f);
  MemRef<float, 0> actual({}, 0.0f);
  initializeInput(source);

  float expected = dyno::regression::reduceLaneGrouped1D(
      source, 0.0f, 0.0f, DYNO_REGRESSION_DYNO_VF,
      [](float lhs, float rhs) { return lhs + rhs; });

  _mlir_ciface_kernel_dyno(&source, &actual);

  dyno::regression::verifyScalar(actual[0], expected, DYNO_REGRESSION_NAME,
                                 kEpsilon);
  return 0;
}

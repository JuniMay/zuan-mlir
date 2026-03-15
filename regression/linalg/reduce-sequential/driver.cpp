#include "RegressionUtils.h"

#include <cstddef>

extern "C" {
void _mlir_ciface_kernel_dyno(MemRef<float, 1> *, MemRef<float, 0> *);
}

namespace {

constexpr int64_t kN = 139;
constexpr float kEpsilon = static_cast<float>(DYNO_REGRESSION_EPSILON);

float makeElement(int64_t index) {
  int64_t lane = index % DYNO_REGRESSION_DYNO_VF;
  if (lane == 0) {
    return 100000000.0f;
  }
  if (lane == 1) {
    return -100000000.0f;
  }
  return 1.0f;
}

void initializeInput(MemRef<float, 1> &source) {
  for (int64_t i = 0; i < kN; ++i) {
    source[i] = makeElement(i);
  }
}

} // namespace

int main() {
  dyno::regression::logReferenceMode("ordered-strict");

  MemRef<float, 1> source({kN}, 0.0f);
  MemRef<float, 0> actual({}, 0.0f);
  initializeInput(source);

  float expected = dyno::regression::reduceOrdered1D(
      source, 0.0f, [](float lhs, float rhs) { return lhs + rhs; });

  _mlir_ciface_kernel_dyno(&source, &actual);

  dyno::regression::verifyScalar(actual[0], expected, DYNO_REGRESSION_NAME,
                                 kEpsilon);
  return 0;
}

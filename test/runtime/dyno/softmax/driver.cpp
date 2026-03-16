#include "RegressionUtils.h"

#include <cmath>
#include <cstddef>

extern "C" {
void _mlir_ciface_kernel_dyno(MemRef<float, 1> *, MemRef<float, 1> *);
}

namespace {

constexpr int64_t kN = 37;
constexpr float kEpsilon = static_cast<float>(DYNO_REGRESSION_EPSILON);

float makeInput(int64_t index) {
  return static_cast<float>(((index * 11) % 17) - 8) * 0.25f +
         static_cast<float>(index % 3) * 0.1f;
}

void initializeInput(MemRef<float, 1> &source) {
  for (int64_t i = 0; i < kN; ++i) {
    source[i] = makeInput(i);
  }
}

void runReference(const MemRef<float, 1> &source, MemRef<float, 1> &dst) {
  float maxValue = source[0];
  for (int64_t i = 1; i < kN; ++i) {
    if (source[i] > maxValue) {
      maxValue = source[i];
    }
  }

  float sum = 0.0f;
  for (int64_t i = 0; i < kN; ++i) {
    dst[i] = std::exp(source[i] - maxValue);
    sum += dst[i];
  }

  for (int64_t i = 0; i < kN; ++i) {
    dst[i] /= sum;
  }
}

} // namespace

int main() {
  dyno::regression::logReferenceMode("ordered-strict");

  MemRef<float, 1> source({kN}, 0.0f);
  MemRef<float, 1> expected({kN}, 0.0f);
  MemRef<float, 1> actual({kN}, 0.0f);

  initializeInput(source);
  runReference(source, expected);
  _mlir_ciface_kernel_dyno(&source, &actual);

  dyno::regression::verifyMemRef(actual, expected, DYNO_REGRESSION_NAME,
                                 kEpsilon);
  return 0;
}

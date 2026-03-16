#include "RegressionUtils.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>

extern "C" {
void _mlir_ciface_kernel_dyno(MemRef<float, 1> *, MemRef<float, 1> *);
}

namespace {

constexpr int64_t kN = 19;

float combineMinimum(float lhs, float rhs) {
  if (std::isnan(lhs) || std::isnan(rhs)) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (lhs == rhs && lhs == 0.0f) {
    return std::signbit(lhs) ? lhs : rhs;
  }
  return lhs < rhs ? lhs : rhs;
}

void initializeInput(MemRef<float, 1> &source) {
  for (int64_t i = 0; i < kN; ++i) {
    source[i] = static_cast<float>((i % 7) - 3) * 0.5f;
  }
  source[3] = -0.0f;
  source[4] = 0.0f;
  source[11] = std::numeric_limits<float>::quiet_NaN();
}

float runReference(const MemRef<float, 1> &source) {
  return dyno::regression::reduceOrdered1D(source, source[0], combineMinimum);
}

} // namespace

int main() {
  dyno::regression::logReferenceMode("minimumf-strict");

  MemRef<float, 1> source({kN}, 0.0f);
  MemRef<float, 1> actual({1}, 0.0f);

  initializeInput(source);
  float expected = runReference(source);
  _mlir_ciface_kernel_dyno(&source, &actual);

  if (!std::isnan(expected) || !std::isnan(actual[0])) {
    std::ostringstream os;
    os << "[verify] " << DYNO_REGRESSION_NAME << " FAIL" << '\n'
       << "  actual_is_nan   = " << std::isnan(actual[0]) << '\n'
       << "  expected_is_nan = " << std::isnan(expected);
    dyno::regression::fail(os.str());
  }

  std::cerr << "[verify] " << DYNO_REGRESSION_NAME << " PASS" << std::endl;
  return 0;
}

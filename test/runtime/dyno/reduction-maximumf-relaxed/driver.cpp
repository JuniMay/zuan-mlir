#include "RegressionUtils.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>

extern "C" {
void _mlir_ciface_kernel_dyno(MemRef<float, 1> *, MemRef<float, 1> *);
}

namespace {

constexpr int64_t kN = 21;

float combineMaximum(float lhs, float rhs) {
  if (std::isnan(lhs) || std::isnan(rhs)) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (lhs == rhs && lhs == 0.0f) {
    return std::signbit(lhs) ? rhs : lhs;
  }
  return lhs > rhs ? lhs : rhs;
}

void initializeInput(MemRef<float, 1> &source) {
  for (int64_t i = 0; i < kN; ++i) {
    source[i] = (i % 3 == 0) ? 0.0f : -0.0f;
  }
}

float runReference(const MemRef<float, 1> &source) {
  return dyno::regression::reduceOrdered1D(source, source[0], combineMaximum);
}

} // namespace

int main() {
  dyno::regression::logReferenceMode("maximumf-relaxed");

  MemRef<float, 1> source({kN}, 0.0f);
  MemRef<float, 1> actual({1}, 0.0f);

  initializeInput(source);
  float expected = runReference(source);
  _mlir_ciface_kernel_dyno(&source, &actual);

  if (std::isnan(actual[0]) || actual[0] != expected ||
      std::signbit(actual[0]) != std::signbit(expected)) {
    std::ostringstream os;
    os << "[verify] " << DYNO_REGRESSION_NAME << " FAIL" << '\n'
       << "  actual        = " << actual[0] << '\n'
       << "  expected      = " << expected << '\n'
       << "  actual_sign   = " << std::signbit(actual[0]) << '\n'
       << "  expected_sign = " << std::signbit(expected);
    dyno::regression::fail(os.str());
  }

  std::cerr << "[verify] " << DYNO_REGRESSION_NAME << " PASS" << std::endl;
  return 0;
}

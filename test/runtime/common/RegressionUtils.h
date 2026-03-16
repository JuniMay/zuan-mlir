#ifndef DYNO_REGRESSION_COMMON_REGRESSIONUTILS_H
#define DYNO_REGRESSION_COMMON_REGRESSIONUTILS_H

#include "common/MemRefUtils.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace dyno {
namespace regression {

[[noreturn]] inline void fail(const std::string &message) {
  std::cerr << message << std::endl;
  std::exit(EXIT_FAILURE);
}

inline void logReferenceMode(const char *mode) {
  std::cerr << "[reference] semantics=" << mode << std::endl;
}

template <typename T>
inline bool equalWithTolerance(T actual, T expected, double epsilon) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::fabs(static_cast<double>(actual) -
                     static_cast<double>(expected)) <= epsilon;
  }
  return actual == expected;
}

template <typename T, size_t N>
std::string formatCoordinates(const MemRef<T, N> &value, int64_t linearIndex) {
  if constexpr (N == 0) {
    return "[]";
  }

  std::vector<int64_t> coordinates(N, 0);
  int64_t remaining = linearIndex;
  for (int64_t dim = static_cast<int64_t>(N) - 1; dim >= 0; --dim) {
    int64_t size = value.getSize(dim);
    coordinates[dim] = remaining % size;
    remaining /= size;
  }

  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < N; ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << coordinates[i];
  }
  os << "]";
  return os.str();
}

template <typename T>
void verifyScalar(T actual, T expected, const std::string &name,
                  double epsilon = 0.0) {
  if (!equalWithTolerance(actual, expected, epsilon)) {
    std::ostringstream os;
    os << "[verify] " << name << " FAIL" << '\n'
       << "  actual   = " << std::setprecision(10) << actual << '\n'
       << "  expected = " << std::setprecision(10) << expected << '\n'
       << "  epsilon  = " << epsilon;
    fail(os.str());
  }

  std::cerr << "[verify] " << name << " PASS" << std::endl;
}

template <typename T, size_t N>
void verifyMemRef(const MemRef<T, N> &actual, const MemRef<T, N> &expected,
                  const std::string &name, double epsilon = 0.0) {
  if (actual.getTotalSize() != expected.getTotalSize()) {
    std::ostringstream os;
    os << "[verify] " << name << " FAIL" << '\n'
       << "  actual-size   = " << actual.getTotalSize() << '\n'
       << "  expected-size = " << expected.getTotalSize();
    fail(os.str());
  }

  for (int64_t linearIndex = 0; linearIndex < actual.getTotalSize();
       ++linearIndex) {
    T actualValue = actual[linearIndex];
    T expectedValue = expected[linearIndex];
    if (!equalWithTolerance(actualValue, expectedValue, epsilon)) {
      std::ostringstream os;
      os << "[verify] " << name << " FAIL" << '\n'
         << "  index    = " << formatCoordinates(actual, linearIndex) << '\n'
         << "  actual   = " << std::setprecision(10) << actualValue << '\n'
         << "  expected = " << std::setprecision(10) << expectedValue << '\n'
         << "  epsilon  = " << epsilon;
      fail(os.str());
    }
  }

  std::cerr << "[verify] " << name << " PASS" << std::endl;
}

template <typename T, typename Combine>
T reduceOrdered1D(const MemRef<T, 1> &source, T init, Combine combine) {
  T acc = init;
  for (int64_t i = 0; i < source.getSize(0); ++i) {
    acc = combine(acc, source[i]);
  }
  return acc;
}

template <typename T, typename Combine>
T reduceLaneGrouped1D(const MemRef<T, 1> &source, T init, T identity, size_t vf,
                      Combine combine) {
  std::vector<T> lanes(vf, identity);
  for (int64_t i = 0; i < source.getSize(0); ++i) {
    lanes[static_cast<size_t>(i) % vf] =
        combine(lanes[static_cast<size_t>(i) % vf], source[i]);
  }

  T acc = init;
  for (T laneAcc : lanes) {
    acc = combine(acc, laneAcc);
  }
  return acc;
}

} // namespace regression
} // namespace dyno

#endif // DYNO_REGRESSION_COMMON_REGRESSIONUTILS_H

#ifndef BENCHMARKS_COMMON_VERIFICATION_H
#define BENCHMARKS_COMMON_VERIFICATION_H

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

template <typename T>
inline void verify(const T* lhs, const T *rhs, size_t N, const std::string &name,
            T epsilon = 0.0001) {
  const std::string PASS = "\033[32mPASS\033[0m";
  const std::string FAIL = "\033[31mFAIL\033[0m";

  std::cerr << name << " ";
  if (!lhs || !rhs) {
    std::cerr << FAIL << " (Null pointer detected)" << std::endl;
    // Benchmarks are only useful if a mismatch fails loudly in CI/QEMU too.
    std::exit(EXIT_FAILURE);
  }

  bool isPass = true;
  for (size_t i = 0; i < N; ++i) {
    if (std::abs(lhs[i] - rhs[i]) > epsilon) {
      std::cerr << FAIL << std::endl;
      std::cerr << "Index " << i << ":\tA=" << std::setprecision(10) << lhs[i]
                << " B=" << std::setprecision(10) << rhs[i] << std::endl;
      // Stop immediately so correctness regressions do not look like a passing
      // benchmark run with only a stderr message.
      std::exit(EXIT_FAILURE);
    }
  }
  if (isPass) {
    std::cerr << PASS << std::endl;
  }
}

#endif // BENCHMARKS_COMMON_VERIFICATION_H

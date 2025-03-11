#ifndef BENCHMARKS_COMMON_VERIFICATION_H
#define BENCHMARKS_COMMON_VERIFICATION_H

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
    return;
  }

  bool isPass = true;
  for (size_t i = 0; i < N; ++i) {
    if (std::abs(lhs[i] - rhs[i]) > epsilon) {
      std::cerr << FAIL << std::endl;
      std::cerr << "Index " << i << ":\tA=" << std::setprecision(10) << lhs[i]
                << " B=" << std::setprecision(10) << rhs[i] << std::endl;
      isPass = false;
      break;
    }
  }
  if (isPass) {
    std::cerr << PASS << std::endl;
  }
}

#endif // BENCHMARKS_COMMON_VERIFICATION_H

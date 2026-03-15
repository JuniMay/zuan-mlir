#include "RegressionUtils.h"

#include <cstddef>

extern "C" {
void _mlir_ciface_kernel_dyno(MemRef<float, 4> *, MemRef<float, 4> *);
}

namespace {

constexpr int64_t kD0 = 2;
constexpr int64_t kD1 = 3;
constexpr int64_t kD2 = 5;
constexpr int64_t kD3 = 7;

float makeInput(int64_t linearIndex) {
  return static_cast<float>((linearIndex * 7) % 19 - 9) * 0.5f;
}

void initializeInput(MemRef<float, 4> &source) {
  for (int64_t i = 0; i < source.getTotalSize(); ++i) {
    source[i] = makeInput(i);
  }
}

void runReference(const MemRef<float, 4> &source, MemRef<float, 4> &dst) {
  for (int64_t i = 0; i < source.getTotalSize(); ++i) {
    dst[i] = source[i] < 0.0f ? 0.0f : source[i];
  }
}

} // namespace

int main() {
  MemRef<float, 4> source({kD0, kD1, kD2, kD3}, 0.0f);
  MemRef<float, 4> expected({kD0, kD1, kD2, kD3}, 0.0f);
  MemRef<float, 4> actual({kD0, kD1, kD2, kD3}, 0.0f);

  initializeInput(source);
  runReference(source, expected);
  _mlir_ciface_kernel_dyno(&source, &actual);

  dyno::regression::verifyMemRef(actual, expected, DYNO_REGRESSION_NAME);
  return 0;
}

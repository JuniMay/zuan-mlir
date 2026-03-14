#include "common/MemRefUtils.h"

// Simple naive implementation with loop interchange.
void naive_matmul(MemRef<float, 2> *a, MemRef<float, 2> *b,
                  MemRef<float, 2> *c) {
  int64_t M = a->getSize(0);
  int64_t N = b->getSize(1);
  int64_t K = a->getSize(1);

  for (int64_t i = 0; i < M; i++) {
    for (int64_t k = 0; k < K; k++) {
      for (int64_t j = 0; j < N; j++) {
        (*c)[i * N + j] += (*a)[i * K + k] * (*b)[k * N + j];
      }
    }
  }
}

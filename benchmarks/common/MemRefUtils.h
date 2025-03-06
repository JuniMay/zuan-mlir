
#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

template <int N> void dropFront(int64_t arr[N], int64_t *res) {
  for (unsigned i = 1; i < N; ++i) {
    *(res + i - 1) = arr[i];
  }
}

template <typename T, size_t N> struct MemRef {
  MemRef(std::vector<size_t> sizes, T init) {
    assert(sizes.size() == N && "dimension mismatch");
    std::copy(sizes.begin(), sizes.end(), this->sizes);
    if (N > 0) {
      strides[N - 1] = 1;
      for (int i = N - 1; i > 0; i--) {
        strides[i - 1] = strides[i] * sizes[i];
      }
    }
    size_t totalSize = std::accumulate(sizes.begin(), sizes.end(), 1,
                                       std::multiplies<size_t>());
    basePtr = static_cast<T *>(malloc(totalSize * sizeof(T)));
    data = basePtr;
    offset = 0;
    std::fill(data, data + totalSize, init);
  }

  MemRef(const MemRef<T, N> &) = delete;

  MemRef(MemRef<T, N> &&other)
      : basePtr(other.basePtr), data(other.data), offset(other.offset) {
    std::swap(sizes, other.sizes);
    std::swap(strides, other.strides);
    other.basePtr = nullptr;
    other.data = nullptr;
  }

  ~MemRef() {
    // std::cout << "FREEING: " << basePtr << std::endl;
    if (basePtr) {
      // std::cout << "FREED: " << basePtr << std::endl;
      free(basePtr);
    }
  }

  T *getData() { return data; }

  /// Given the linearized index, return the reference to the element.
  T &operator[](int64_t idx) { return data[offset + idx]; }

  int64_t getTotalSize() const {
    return std::accumulate(sizes, sizes + N, 1, std::multiplies<int64_t>());
  }
  void fill(T val) {
    size_t totalSize = getTotalSize();
    std::fill(data, data + totalSize, val);
  }

  void verify(const MemRef<T, N> &other, const std::string &name,
              T epsilon = 0.0001) const {
    const std::string PASS = "\033[32mPASS\033[0m";
    const std::string FAIL = "\033[31mFAIL\033[0m";

    std::cout << name << " ";
    if (!data || !other.data) {
      std::cout << FAIL << " (Null pointer detected)" << std::endl;
      return;
    }

    auto totalSize = getTotalSize();
    assert(totalSize == other.getTotalSize() && "size mismatch");
    bool isPass = true;
    for (int i = 0; i < totalSize; ++i) {
      if (std::abs(data[i] - other.data[i]) > epsilon) {
        std::cout << FAIL << std::endl;
        std::cout << "Index " << i << ":\tA=" << std::setprecision(10)
                  << data[i] << " B=" << std::setprecision(10) << other.data[i]
                  << std::endl;
        isPass = false;
        break;
      }
    }
    if (isPass) {
      std::cout << PASS << std::endl;
    }
  }

  int64_t getSize(size_t idx) const { return sizes[idx]; }

private:
  T *basePtr = nullptr;
  T *data = nullptr;

  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

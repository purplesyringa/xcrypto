#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdint.h>
#include <string_view>
#include <vector>

#ifndef EXPORT
#define EXPORT
#endif

namespace EXPORT bigint {

constexpr void ensure(bool cond) {
  if (!cond) {
    abort();
  }
}

void memzero64(uint64_t *data, size_t count) {
  asm volatile("rep stosq" : "+D"(data), "+c"(count) : "a"(0) : "memory");
}

void memcpy64(uint64_t *dst, const uint64_t *src, size_t count) {
  asm volatile("rep movsq" : "+D"(dst), "+S"(src), "+c"(count) : : "memory");
}

struct EXPORT with_base {
  uint64_t base;
};

class ConstSpan;

class SmallVec {
  static constexpr size_t INLINE_STORAGE_SIZE = 8;

  uint64_t *_begin;
  size_t _size;
  size_t _capacity;
  uint64_t _inline_storage[INLINE_STORAGE_SIZE];

  void increase_capacity_to(size_t new_capacity) {
    uint64_t *new_begin = new uint64_t[new_capacity];
    memcpy64(new_begin, _begin, _size);
    if (_begin != _inline_storage) {
      delete[] _begin;
    }
    _begin = new_begin;
    _capacity = new_capacity;
  }

public:
  SmallVec()
      : _begin(_inline_storage), _size(0), _capacity(INLINE_STORAGE_SIZE) {}
  SmallVec(std::initializer_list<uint64_t> list) {
    if (list.size() <= INLINE_STORAGE_SIZE) {
      _begin = _inline_storage;
      _capacity = INLINE_STORAGE_SIZE;
    } else {
      _begin = new uint64_t[list.size() + 1];
      _capacity = list.size() + 1;
    }
    std::copy(list.begin(), list.end(), _begin);
    _size = list.size();
  }
  SmallVec(const uint64_t *data, size_t size) {
    if (size <= INLINE_STORAGE_SIZE) {
      _begin = _inline_storage;
      _capacity = INLINE_STORAGE_SIZE;
    } else {
      _begin = new uint64_t[size + 1];
      _capacity = size + 1;
    }
    memcpy64(_begin, data, size);
    _size = size;
  }

  SmallVec(ConstSpan rhs);
  SmallVec(const SmallVec &rhs);
  SmallVec(SmallVec &&rhs) {
    if (rhs._begin == rhs._inline_storage) {
      _begin = _inline_storage;
      memcpy64(_begin, rhs._begin, rhs._size);
    } else {
      _begin = rhs._begin;
    }
    _size = rhs._size;
    _capacity = rhs._capacity;
    rhs._begin = rhs._inline_storage;
    rhs._size = 0;
    rhs._capacity = 0;
  }

  SmallVec &operator=(ConstSpan rhs);
  SmallVec &operator=(SmallVec &&rhs) {
    if (_begin != _inline_storage) {
      delete[] _begin;
    }
    if (rhs._begin == rhs._inline_storage) {
      _begin = _inline_storage;
      _capacity = INLINE_STORAGE_SIZE;
      _size = rhs._size;
      memcpy64(_begin, rhs._begin, rhs._size);
    } else {
      _begin = rhs._begin;
      _size = rhs._size;
      _capacity = rhs._capacity;
    }
    rhs._begin = rhs._inline_storage;
    rhs._size = 0;
    rhs._capacity = 0;
    return *this;
  }

  ~SmallVec() {
    if (_begin != _inline_storage) {
      delete[] _begin;
    }
  }

  void increase_size(size_t new_size) {
    if (_capacity < new_size) {
      increase_capacity_to(new_size);
    }
    _size = new_size;
  }
  void increase_size_zerofill(size_t new_size) {
    if (_capacity < new_size) {
      increase_capacity_to(new_size);
    }
    memzero64(_begin + _size, new_size - _size);
    _size = new_size;
  }
  void ensure_size(size_t new_size) {
    if (new_size <= _size) {
      return;
    }
    increase_size(new_size);
  }
  void ensure_size_zerofill(size_t new_size) {
    if (new_size <= _size) {
      return;
    }
    increase_size_zerofill(new_size);
  }
  void set_size(size_t size) { _size = size; }

  uint64_t &operator[](size_t i) { return _begin[i]; }
  const uint64_t &operator[](size_t i) const { return _begin[i]; }

  size_t size() const { return _size; }
  bool empty() const { return _size == 0; }
  uint64_t *data() { return _begin; }
  const uint64_t *data() const { return _begin; }

  void push_back(uint64_t x) {
    if (_size == _capacity) {
      increase_capacity_to(_capacity * 2 + 1);
    }
    _begin[_size++] = x;
  }
  void pop_back() { _size--; }
  void clear_dealloc() {
    if (_begin != _inline_storage) {
      delete[] _begin;
    }
    _begin = _inline_storage;
    _size = 0;
    _capacity = 0;
  }

  bool operator==(const SmallVec &rhs) const {
    return _size == rhs._size && std::equal(_begin, _begin + _size, rhs._begin);
  }

  uint64_t &back() { return _begin[_size - 1]; }
  const uint64_t &back() const { return _begin[_size - 1]; }

  uint64_t *begin() { return _begin; }
  uint64_t *end() { return _begin + _size; }
  const uint64_t *begin() const { return _begin; }
  const uint64_t *end() const { return _begin + _size; }

  std::reverse_iterator<uint64_t *> rbegin() {
    return std::make_reverse_iterator(_begin + _size);
  }
  std::reverse_iterator<uint64_t *> rend() {
    return std::make_reverse_iterator(_begin);
  }
  std::reverse_iterator<const uint64_t *> rbegin() const {
    return std::make_reverse_iterator(_begin + _size);
  }
  std::reverse_iterator<const uint64_t *> rend() const {
    return std::make_reverse_iterator(_begin);
  }
};

class Span {
  uint64_t *_begin;
  size_t _size;

public:
  Span() : _begin(nullptr), _size(0) {}
  Span(uint64_t *data, size_t size) : _begin(data), _size(size) {}
  Span(SmallVec &vec) : _begin(vec.begin()), _size(vec.size()) {}
  void set_size(size_t size) { _size = size; }

  uint64_t &operator[](size_t i) { return _begin[i]; }
  const uint64_t &operator[](size_t i) const { return _begin[i]; }

  size_t size() const { return _size; }
  bool empty() const { return _size == 0; }
  uint64_t *data() { return _begin; }
  const uint64_t *data() const { return _begin; }

  void pop_back() { _size--; }

  bool operator==(const Span &rhs) const {
    return _size == rhs._size && std::equal(_begin, _begin + _size, rhs._begin);
  }

  uint64_t &back() { return _begin[_size - 1]; }
  uint64_t back() const { return _begin[_size - 1]; }

  uint64_t *begin() { return _begin; }
  uint64_t *end() { return _begin + _size; }
  const uint64_t *begin() const { return _begin; }
  const uint64_t *end() const { return _begin + _size; }

  std::reverse_iterator<uint64_t *> rbegin() {
    return std::make_reverse_iterator(_begin + _size);
  }
  std::reverse_iterator<uint64_t *> rend() {
    return std::make_reverse_iterator(_begin);
  }
  std::reverse_iterator<const uint64_t *> rbegin() const {
    return std::make_reverse_iterator(_begin + _size);
  }
  std::reverse_iterator<const uint64_t *> rend() const {
    return std::make_reverse_iterator(_begin);
  }
};

class ConstSpan {
  const uint64_t *_begin;
  size_t _size;

public:
  ConstSpan() : _begin(nullptr), _size(0) {}
  ConstSpan(const uint64_t *data, size_t size) : _begin(data), _size(size) {}
  ConstSpan(const SmallVec &vec) : _begin(vec.begin()), _size(vec.size()) {}
  ConstSpan(Span span) : _begin(span.begin()), _size(span.size()) {}
  void set_size(size_t size) { _size = size; }

  const uint64_t &operator[](size_t i) const { return _begin[i]; }

  size_t size() const { return _size; }
  bool empty() const { return _size == 0; }
  const uint64_t *data() const { return _begin; }

  void pop_back() { _size--; }

  bool operator==(const ConstSpan &rhs) const {
    return _size == rhs._size && std::equal(_begin, _begin + _size, rhs._begin);
  }

  const uint64_t &back() const { return _begin[_size - 1]; }

  const uint64_t *begin() const { return _begin; }
  const uint64_t *end() const { return _begin + _size; }

  std::reverse_iterator<const uint64_t *> rbegin() const {
    return std::make_reverse_iterator(_begin + _size);
  }
  std::reverse_iterator<const uint64_t *> rend() const {
    return std::make_reverse_iterator(_begin);
  }
};

SmallVec::SmallVec(ConstSpan rhs) {
  if (rhs.size() <= INLINE_STORAGE_SIZE) {
    _begin = _inline_storage;
    _capacity = INLINE_STORAGE_SIZE;
  } else {
    _begin = new uint64_t[rhs.size() + 1];
    _capacity = rhs.size() + 1;
  }
  memcpy64(_begin, rhs.data(), rhs.size());
  _size = rhs.size();
}

SmallVec::SmallVec(const SmallVec &rhs)
    : SmallVec(static_cast<ConstSpan>(rhs)) {}

SmallVec &SmallVec::operator=(ConstSpan rhs) {
  if (rhs.size() > _capacity) {
    if (_begin != _inline_storage) {
      delete[] _begin;
    }
    _begin = new uint64_t[rhs.size() + 1];
    _capacity = rhs.size() + 1;
  }
  memcpy64(_begin, rhs.data(), rhs.size());
  _size = rhs.size();
  return *this;
}

class Ref;
class ConstRef;

class EXPORT BigInt {
  Ref slice(size_t l);
  Ref slice(size_t l, size_t size);
  ConstRef slice(size_t l) const;
  ConstRef slice(size_t l, size_t size) const;

  BigInt(SmallVec data);

public:
  SmallVec data;

  BigInt();

  BigInt(__uint128_t value);
  BigInt(uint64_t value);
  BigInt(int value);

  template <typename List, typename = decltype(std::declval<List>().begin())>
  BigInt(List &&list, with_base base);
  BigInt(std::string_view s, with_base base = {10});
  BigInt(const char *s, with_base base = {10});

  BigInt(ConstRef rhs);
  BigInt(const BigInt &rhs);
  BigInt(BigInt &&rhs);

  BigInt &operator=(ConstRef rhs);
  BigInt &operator=(BigInt &&rhs);

  BigInt &operator+=(ConstRef rhs);
  BigInt &operator-=(ConstRef rhs);
  BigInt &operator+=(const BigInt &rhs);
  BigInt &operator-=(const BigInt &rhs);

  BigInt &operator++();
  BigInt operator++(int);

  BigInt &operator--();
  BigInt operator--(int);

  uint32_t divmod_inplace(uint32_t rhs);
  void divide_inplace_whole(uint64_t rhs);

  void _normalize();
  void _normalize_nonzero();

  bool halve();

  friend class ConstRef;
};

class Ref {
public:
  Span data;

  Ref(Span data) : data(data) {}
  Ref(BigInt &bigint) : data(bigint.data) {}
  Ref(BigInt &&bigint) : data(bigint.data) {}

  Ref slice(size_t l) { return {Span{data.data() + l, data.size() - l}}; }
  Ref slice(size_t l, size_t size) { return {Span{data.data() + l, size}}; }
  ConstRef slice(size_t l) const;
  ConstRef slice(size_t l, size_t size) const;

  Ref normalized() {
    Span tmp = data;
    while (!tmp.empty() && tmp.back() == 0) {
      tmp.pop_back();
    }
    return {tmp};
  }
};

class ConstRef {
public:
  ConstSpan data;

  ConstRef(ConstSpan data) : data(data) {}
  ConstRef(const BigInt &bigint) : data(bigint.data) {}
  ConstRef(Ref ref) : data(ref.data) {}

  ConstRef slice(size_t l) const {
    return {ConstSpan{data.data() + l, data.size() - l}};
  }
  ConstRef slice(size_t l, size_t size) const {
    return {ConstSpan{data.data() + l, size}};
  }

  explicit operator BigInt() const {
    return {SmallVec{data.data(), data.size()}};
  }

  ConstRef normalized() {
    ConstSpan tmp = data;
    while (!tmp.empty() && tmp.back() == 0) {
      tmp.pop_back();
    }
    return {tmp};
  }
};

ConstRef Ref::slice(size_t l) const {
  return static_cast<ConstRef>(*this).slice(l);
}
ConstRef Ref::slice(size_t l, size_t size) const {
  return static_cast<ConstRef>(*this).slice(l, size);
}

Ref BigInt::slice(size_t l) { return static_cast<Ref>(*this).slice(l); }
Ref BigInt::slice(size_t l, size_t size) {
  return static_cast<Ref>(*this).slice(l, size);
}
ConstRef BigInt::slice(size_t l) const {
  return static_cast<ConstRef>(*this).slice(l);
}
ConstRef BigInt::slice(size_t l, size_t size) const {
  return static_cast<ConstRef>(*this).slice(l, size);
}

void BigInt::_normalize() {
  while (!data.empty() && data.back() == 0) {
    data.pop_back();
  }
}
void BigInt::_normalize_nonzero() {
  while (data.back() == 0) {
    data.pop_back();
  }
}

bool BigInt::halve() {
  // rcr/rcl are not going to help performance here, neither will SIMD -- prefer shrd
  size_t size = data.size();
  if (size == 0) {
    return 0;
  } else if (size == 1) {
    bool carry = data[0] & 1;
    data[0] >>= 1;
    return carry;
  }

  uint64_t prev = data[0];
  bool carry = prev & 1;

  size_t i = 1;
  while (i <= size - 2) {
    uint64_t word1 = data[i];
    uint64_t word2 = data[i + 1];
    data[i - 1] = ((__uint128_t{word1} << 64) | prev) >> 1;
    data[i] = ((__uint128_t{word2} << 64) | word1) >> 1;
    prev = word2;
    i += 2;
  }

  if (i < size) {
    uint64_t word = data[i];
    data[i - 1] = ((__uint128_t{word} << 64) | prev) >> 1;
    prev = word;
  }

  data.back() = prev >> 1;
  _normalize();
  return carry;
}

void add_to(Ref lhs, ConstRef rhs) {
  if (__builtin_expect(rhs.data.empty(), 0)) {
    return;
  }

  size_t offset = 0;
  uint64_t value1, value2;
  size_t unrolled_loop_count = rhs.data.size() / 4;
  size_t left_loop_count = rhs.data.size() % 4;
  if (left_loop_count == 0) {
    unrolled_loop_count--;
    left_loop_count = 4;
  }
  asm volatile(
      "test %[unrolled_loop_count], %[unrolled_loop_count];"
      "jz 2f;"
      "1:"
      "mov (%[rhs_data_ptr],%[offset]), %[value1];"
      "mov 0x8(%[rhs_data_ptr],%[offset]), %[value2];"
      "adc %[value1], (%[data_ptr],%[offset]);"
      "adc %[value2], 0x8(%[data_ptr],%[offset]);"
      "mov 0x10(%[rhs_data_ptr],%[offset]), %[value1];"
      "mov 0x18(%[rhs_data_ptr],%[offset]), %[value2];"
      "adc %[value1], 0x10(%[data_ptr],%[offset]);"
      "adc %[value2], 0x18(%[data_ptr],%[offset]);"
      "lea 0x20(%[offset]), %[offset];"
      "dec %[unrolled_loop_count];"
      "jnz 1b;"
      "2:"
      "mov (%[rhs_data_ptr],%[offset]), %[value1];"
      "adc %[value1], (%[data_ptr],%[offset]);"
      "lea 0x8(%[offset]), %[offset];"
      "dec %[left_loop_count];"
      "jnz 2b;"
      "jnc 4f;"
      "3:"
      "addq $1, (%[data_ptr],%[offset]);"
      "lea 0x8(%[offset]), %[offset];"
      "jc 3b;"
      "4:"
      : [offset] "+r"(offset), [value1] "=&r"(value1), [value2] "=&r"(value2),
        [unrolled_loop_count] "+r"(unrolled_loop_count),
        [left_loop_count] "+r"(left_loop_count)
      : [data_ptr] "r"(lhs.data.data()), [rhs_data_ptr] "r"(rhs.data.data())
      : "flags", "memory");
}

inline constexpr int FFT_CUTOFF = 14;
inline constexpr int CT8_CUTOFF = 15;
inline constexpr int FFT_MIN =
    FFT_CUTOFF - 1; // -1 due to real-fft size halving optimization
inline constexpr int FFT_MAX = 19;

using Complex = double[2];

inline auto twiddles_bitreversed = new (std::align_val_t(32)) Complex[8];
inline int twiddles_n_pow = 0;

uint32_t bitreverse(uint32_t k, int n_pow) {
  k <<= 32 - n_pow;
  k = ((k & 0x55555555) << 1 ) | ((k >> 1 ) & 0x55555555);
  k = ((k & 0x33333333) << 2 ) | ((k >> 2 ) & 0x33333333);
  k = ((k & 0x0f0f0f0f) << 4 ) | ((k >> 4 ) & 0x0f0f0f0f);
  k = ((k & 0x00ff00ff) << 8 ) | ((k >> 8 ) & 0x00ff00ff);
  k = ((k & 0x0000ffff) << 16) | ((k >> 16) & 0x0000ffff);
  return k;
}

void ensure_twiddle_factors(int want_n_pow) {
  static constexpr double PI = 3.1415926535897931;

  if (twiddles_n_pow == 0) {
    // bitreversed k | k | n | 2 pi k / n | cos      | sin
    // 0             | 0 | 1 | 0          | 1        | 0
    // 1             | 1 | 2 | pi         | -1       | 0
    // 2             | 1 | 4 | pi/2       | 0        | 1
    // 3             | 3 | 4 | 3pi/2      | 0        | -1
    // 4             | 1 | 8 | pi/4       | 1/sqrt2  | 1/sqrt2
    // 5             | 5 | 8 | 5pi/4      | -1/sqrt2 | -1/sqrt2
    // 6             | 3 | 8 | 3pi/4      | -1/sqrt2 | 1/sqrt2
    // 7             | 7 | 8 | 7pi/4      | 1/sqrt2  | -1/sqrt2
    const double r = sqrt(.5);
    const double values[] = {1, 0, -1, 0, 0, 1, 0, -1};
    memcpy64(
      reinterpret_cast<uint64_t*>(twiddles_bitreversed),
      reinterpret_cast<const uint64_t*>(values),
      8
    );
    twiddles_n_pow = 2;
  }

  while (twiddles_n_pow < want_n_pow) {
    twiddles_n_pow++;
    size_t n = size_t{1} << twiddles_n_pow;
    auto new_twiddles_bitreversed = new (std::align_val_t(32)) Complex[n];
    memcpy64(reinterpret_cast<uint64_t *>(new_twiddles_bitreversed),
             reinterpret_cast<uint64_t *>(twiddles_bitreversed), n);
    ::operator delete[](twiddles_bitreversed, std::align_val_t(32));
    twiddles_bitreversed = new_twiddles_bitreversed;
    double coeff = 2 * PI / static_cast<double>(n);
    for (size_t k = n / 8; k < n / 4; k++) {
      double c = std::cos(coeff * bitreverse(4 * k, twiddles_n_pow));
      twiddles_bitreversed[4 * k][0] = c;
      twiddles_bitreversed[4 * k + 1][0] = -c;
      twiddles_bitreversed[4 * k + 2][1] = c;
      twiddles_bitreversed[4 * k + 3][1] = -c;
    }
  }
}

int get_fft_n_pow(ConstRef lhs, ConstRef rhs) {
  return 64 - __builtin_clzll((lhs.data.size() + rhs.data.size() - 1) * 4 - 1);
}

void fft_dif(Complex *a, int n_pow) {
  // FFT-DIF is defined by
  //     FFT-DIF[P]_k = FFT[P]_{bitreverse(k)}
  // For
  //     P(x) = E(x^2) + x O(x^2)
  // we have
  //     FFT[P]_k = FFT[E]_k + w_n^k FFT[O]_k.
  // For k = 0 to n/2 - 1, we thus obtain
  //     FFT[P]_k       = FFT[E]_k + w_n^k FFT[O]_k
  //     FFT[P]_{k+n/2} = FFT[E]_k - w_n^k FFT[O]_k,
  // and thus
  //     FFT-DIF[P]_{2k}   = FFT-DIF[E]_k + w_n^{bitreverse(k)} FFT-DIF[O]_k
  //     FFT-DIF[P]_{2k+1} = FFT-DIF[E]_k - w_n^{bitreverse(k)} FFT-DIF[O]_k,
  // where bitreverse is of length n (not n/2).
  //
  // FFT-DIF of an array A can thus be computed recursively as follows:
  //     def FFT-DIF(A):
  //         FFT-DIF(A[::2])
  //         FFT-DIF(A[1::2])
  //         for k in range(n/2):
  //             e, o, w = A[2k], A[2k+1], W[k]
  //             A[2k] = e + w * o
  //             A[2k+1] = e - w * o
  // Or iteratively as follows:
  //     def FFT-DIF(A):
  //         for step in range(log_2 n - 1, -1, -1):
  //             for k in range(0, n / 2^{step+1}):
  //                 for j in range(2k * 2^step, (2k+1) * 2^step):
  //                     e, o, w = A[j], A[j+2^step], W[k]
  //                     A[j] = e + w * o
  //                     A[j+2^step] = e - w * o

  for (int step = n_pow - 1; step >= 0; step--) {
    for (size_t k = 0; k < (size_t{1} << (n_pow - step - 1)); k++) {
      for (size_t j = (2 * k) << step; j < ((2 * k + 1) << step); j++) {
        double *e = a[j];
        double *o = a[j + (size_t{1} << step)];
        double *w = twiddles_bitreversed[k];
        double e_real = a[j][0];
        double e_imag = a[j][1];
        double wo_real = w[0] * o[0] - w[1] * o[1];
        double wo_imag = w[0] * o[1] + w[1] * o[0];
        e[0] = e_real + wo_real;
        e[1] = e_imag + wo_imag;
        o[0] = e_real - wo_real;
        o[1] = e_imag - wo_imag;
      }
    }
  }
}

void fft_dit(Complex *a, int n_pow) {
  // FFT-DIT is defined by
  //     FFT-DIT[P] = FFT[P'],
  // where
  //     P'_j = P_{bitreverse(j)}
  // Let
  //     P(x) = p_0 + p_1 x + p_2 x^2 + ... + p_{n-1} x^{n-1}
  //     L(x) = p_0 + p_1 x + ... + p_{n/2-1} x^{n/2-1}
  //     H(x) = p_{n/2} + p_{n/2+1} x + ... + p_{n-1} x^{n/2-1},
  // then
  //     P(x) = L(x) + x^{n/2} H(x)
  // and
  //     P'(x) = L'(x^2) + x H'(x^2)
  // Hence from
  //     FFT[P']_k       = FFT[L']_k + w_n^k FFT[H']_k
  //     FFT[P']_{k+n/2} = FFT[L']_k - w_n^k FFT[H']_k
  // we obtain
  //     FFT-DIT[P]_k       = FFT-DIT[L]_k + w_n^{bitreverse(k)} FFT-DIT[H]_k
  //     FFT-DIT[P]_{k+n/2} = FFT-DIT[L]_k - w_n^{bitreverse(k)} FFT-DIT[H]_k,
  // where bitreverse is of length n (not n/2).
  //
  // FFT-DIT of an array A can thus be computed recursively as follows:
  //     def FFT-DIT(A):
  //         FFT-DIT(A[:n/2])
  //         FFT-DIT(A[n/2:])
  //         for k in range(n/2):
  //             l, h, w = A[k], A[k+n/2], W[k]
  //             A[k] = l + w * h
  //             A[k+n/2] = l - w * h
  // Or iteratively as follows:
  //     def FFT-DIT(A):
  //         for step in range(log_2 n - 1, -1, -1):
  //             for k in range(0, n / 2^{step+1}):
  //                 for j in range(2k * 2^step, (2k+1) * 2^step):
  //                     l, h, w = A[j], A[j+2^step], W[j]
  //                     A[j] = l + w * h
  //                     A[j+2^step] = l - w * h

  for (int step = n_pow - 1; step >= 0; step--) {
    for (size_t k = 0; k < (size_t{1} << (n_pow - step - 1)); k++) {
      for (size_t j = (2 * k) << step; j < ((2 * k + 1) << step); j++) {
        double *l = a[j];
        double *h = a[j + (size_t{1} << step)];
        double *w = twiddles_bitreversed[j];
        double l_real = a[j][0];
        double l_imag = a[j][1];
        double wh_real = w[0] * h[0] - w[1] * h[1];
        double wh_imag = w[0] * h[1] + w[1] * h[0];
        l[0] = l_real + wh_real;
        l[1] = l_imag + wh_imag;
        h[0] = l_real - wh_real;
        h[1] = l_imag - wh_imag;
      }
    }
  }
}

Complex* mul_fft_transform_input(ConstRef input, int n_pow) {
  size_t n = size_t{1} << n_pow;

  // CT4 reads out of bounds, so add + 2
  Complex *input_fft = new (std::align_val_t(32)) Complex[n / 2 + 2];

  // Split into 16-bit words
  const uint16_t *data = reinterpret_cast<const uint16_t *>(input.data.data());
  for (size_t i = 0; i < input.data.size() * 4; i++) {
    input_fft[i][0] = static_cast<double>(data[i]);
    input_fft[i][1] = 0;
  }
  for (size_t i = input.data.size() * 4; i < n / 2 + 2; i++) {
    input_fft[i][0] = 0;
    input_fft[i][1] = 0;
  }

  fft_dif(input_fft, n_pow - 1);

  return input_fft;
}

Complex* mul_fft_middle_end(Complex *lhs_fft_dif, Complex *rhs_fft_dif, int n_pow) {
  size_t n = size_t{1} << n_pow;

  // CT4 reads out of bounds, so add + 2
  Complex *prod_fft_dif = new (std::align_val_t(32)) Complex[n / 2 + 2];

  auto handle_iteration = [&](size_t k, size_t k_complement) {
    double *lhs_k = lhs_fft_dif[k];
    double *rhs_k = rhs_fft_dif[k];
    double *lhs_k_complement = lhs_fft_dif[k_complement];
    double *rhs_k_complement = rhs_fft_dif[k_complement];
    double *twiddles_bitreversed_k = twiddles_bitreversed[k];
    double *result_k = prod_fft_dif[k];
    // lhs_k * rhs_k - (1 + twiddles_bitreversed_k[k]) * (lhs_k - lhs_k_complement.conj()) * (rhs_k - rhs_k_complement.conj()) / 4

    double a_real = lhs_k[0] - lhs_k_complement[0];
    double a_imag = lhs_k[1] + lhs_k_complement[1];
    double b_real = rhs_k[0] - rhs_k_complement[0];
    double b_imag = rhs_k[1] + rhs_k_complement[1];
    double c_real = a_real * b_real - a_imag * b_imag;
    double c_imag = a_real * b_imag + a_imag * b_real;

    result_k[0] = (
      (lhs_k[0] * rhs_k[0] - lhs_k[1] * rhs_k[1])
      - ((1 + twiddles_bitreversed_k[0]) * c_real - twiddles_bitreversed_k[1] * c_imag) / 4
    );
    result_k[1] = (
      (lhs_k[0] * rhs_k[1] + lhs_k[1] * rhs_k[0])
      - ((1 + twiddles_bitreversed_k[0]) * c_imag + twiddles_bitreversed_k[1] * c_real) / 4
    );
  };

  handle_iteration(0, 0);
  for (int j = 0; j < n_pow - 1; j++) {
    for (size_t k = (size_t{1} << j); k < (size_t{2} << j); k++) {
      size_t k_complement = (size_t{3} << j) - 1 - k;
      handle_iteration(k, k_complement);
    }
  }

  return prod_fft_dif;
}

BigInt mul_fft_transform_output(Complex *prod_fft_dif, int n_pow) {
  size_t n = size_t{1} << n_pow;

  fft_dit(prod_fft_dif, n_pow - 1);

  BigInt result;
  result.data.increase_size(n / 4);

  uint64_t carry = 0;

  for (size_t k = 0; k < n / 4; k++) {
    size_t minus_k1 = k == 0 ? 0 : n / 2 - k * 2;
    size_t minus_k2 = n / 2 - 1 - k * 2;

    uint64_t word1 = static_cast<uint64_t>(prod_fft_dif[minus_k1][0] * (static_cast<double>(2) / n) + 0.5);
    uint64_t word2 = static_cast<uint64_t>(prod_fft_dif[minus_k1][1] * (static_cast<double>(2) / n) + 0.5);
    uint64_t word3 = static_cast<uint64_t>(prod_fft_dif[minus_k2][0] * (static_cast<double>(2) / n) + 0.5);
    uint64_t word4 = static_cast<uint64_t>(prod_fft_dif[minus_k2][1] * (static_cast<double>(2) / n) + 0.5);

    __uint128_t tmp = (
      carry
      + word1
      + (__uint128_t{word2} << 16)
      + (__uint128_t{word3} << 32)
      + (__uint128_t{word4} << 48)
    );

    result.data[k] = static_cast<uint64_t>(tmp);
    carry = static_cast<uint64_t>(tmp >> 64);
  }

  if (carry > 0) {
    result.data.push_back(carry);
  } else {
    result._normalize_nonzero();
  }

  return result;
}

BigInt mul_fft(ConstRef lhs, ConstRef rhs) {
  // We use a trick to compute n-sized FFT of real-valued input using a single n/2-sized FFT as
  // follows.
  //
  // Suppose
  //     P(x) = p_0 + p_1 x + p_2 x^2 + ... + p_{n-1} x^{n-1}
  // has real coefficients. Let
  //     E(x) = p_0 + p_2 x + p_4 x^2 + ... + p_{n-2} x^{n/2-1}
  //     O(x) = p_1 + p_3 x + p_5 x^2 + ... + p_{n-1} x^{n/2-1},
  // then
  //     P(x) = E(x^2) + x O(x^2).
  // For
  //     w_n^k = e^{2pi i k / n},
  // we have
  //     P(w_n^k) = E(w_n^{2k}) + w_n^k O(w_n^{2k}) = E(w_{n/2}^k) + w_n^k O(w_{n/2}^k).
  // Therefore,
  //     FFT[P]_k = FFT[E]_k + w_n^k FFT[O]_k,
  // where FFT is assumed to output a cyclic array.
  //
  // On the other hand, let
  //     Q(x) = E(x) + i O(x).
  // By linearity of Fourier transform, we have
  //     FFT[Q]_k = FFT[E]_k + i FFT[O]_k.
  // As E(x) and O(x) have real coefficients, we have conj(E(x)) = E(conj(x)) and an identical
  // formula for O(x). Therefore,
  //     FFT[E]_{-k} = conj(FFT[E]_k)
  //     FFT[O]_{-k} = conj(FFT[O]_k),
  // which implies
  //     FFT[E]_{-k} + i FFT[O]_{-k} = conj(FFT[E]_k) + i conj(FFT[O]_k)
  // Rewriting this as
  //     conj(FFT[Q]_{-k}) = FFT[E]_k - i FFT[O]_k,
  // we obtain the following formulae:
  //     FFT[E]_k = (FFT[Q]_k + conj(FFT[Q]_{-k})) / 2
  //     FFT[O]_k = (FFT[Q]_k - conj(FFT[Q]_{-k})) / 2i
  //
  // Substituting this into the formula for FFT[P], we get
  //     FFT[P]_k = (FFT[Q]_k + conj(FFT[Q]_{-k}) - i w_n^k (FFT[Q]_k - conj(FFT[Q]_{-k}))) / 2
  //              = ((1 - i w_n^k) FFT[Q]_k + (1 + i w_n^k) conj(FFT[Q]_{-k})) / 2
  //
  // Clearly, FFT[Q] can be computed from FFT[P] as well. As
  //     FFT[P]_k = FFT[E]_k + w_n^k FFT[O]_k,
  // we also have
  //     FFT[P]_{k+n/2} = FFT[E]_{k+n/2} + w_n^{k+n/2} FFT[O]_{k+n/2},
  // but both FFT[E] and FFT[O] have a period of only n/2, so
  //     FFT[P]_{k+n/2} = FFT[E]_k - w_n^k FFT[O]_k,
  // which implies
  //     FFT[E]_k = (FFT[P]_k + FFT[P]_{k+n/2}) / 2
  //     FFT[O]_k = (FFT[P]_k - FFT[P]_{k+n/2}) w_n^{-k} / 2,
  // and therefore
  //     FFT[Q]_k = (FFT[P]_k + FFT[P]_{k+n/2} + i w_n^{-k} (FFT[P]_k - FFT[P]_{k+n/2})) / 2
  //
  // Note how similar this forward formula looks to the backward formula.
  //
  // Suppose now we have two polynomials P1(x) and P2(x) and want to compute their product
  // Pr(x) = P1(x) * P2(x) by doing the following steps:
  // 1. Transform P1(x) and P2(x) to Q1(x) and Q2(x)
  // 2. Compute FFT[Q1] and FFT[Q2]
  // 3. Transform FFT[Q1] and FFT[Q2] to FFT[P1] and FFT[P2]
  // 4. Obtain FFT[Pr] by pointwise multiplication
  // 5. Transform FFT[Pr] to FFT[Qr]
  // 6. Compute IFFT[FFT[Qr]] = Qr(x)
  // 7. Transform Qr(x) to Pr(x)
  //
  // For steps 3-4, we have
  //     FFT[Pr]_k = FFT[P1]_k * FFT[P2]_k
  //               = ((1 - i w_n^k) FFT[Q1]_k + (1 + i w_n^k) conj(FFT[Q1]_{-k})) / 2 *
  //                 ((1 - i w_n^k) FFT[Q2]_k + (1 + i w_n^k) conj(FFT[Q2]_{-k})) / 2
  //               = (
  //                     (1 - i w_n^k)^2 FFT[Q1]_k FFT[Q2]_k
  //                     + (1 + w_{n/2}^k) FFT[Q1]_k conj(FFT[Q2]_{-k})
  //                     + (1 + w_{n/2}^k) conj(FFT[Q1]_{-k}) FFT[Q2]_k
  //                     + (1 + i w_n^k)^2 conj(FFT[Q1]_{-k}) conj(FFT[Q2]_{-k})
  //                 ) / 4,
  // which also translates to
  //     FFT[Pr]_{k+n/2} = (
  //                           (1 + i w_n^k)^2 FFT[Q1]_k FFT[Q2]_k
  //                           + (1 + w_{n/2}^k) FFT[Q1]_k conj(FFT[Q2]_{-k})
  //                           + (1 + w_{n/2}^k) conj(FFT[Q1]_{-k}) FFT[Q2]_k
  //                           + (1 - i w_n^k)^2 conj(FFT[Q1]_{-k}) conj(FFT[Q2]_{-k})
  //                       ) / 4
  // Therefore, for the sum and difference of the above we obtain
  //     FFT[Pr]_k + FFT[Pr]_{k+n/2} = (
  //                                       (1 - w_{n/2}^k) FFT[Q1]_k FFT[Q2]_k
  //                                       + (1 + w_{n/2}^k) FFT[Q1]_k conj(FFT[Q2]_{-k})
  //                                       + (1 + w_{n/2}^k) conj(FFT[Q1]_{-k}) FFT[Q2]_k
  //                                       + (1 - w_{n/2}^k) conj(FFT[Q1]_{-k}) conj(FFT[Q2]_{-k})
  //                                   ) / 2,
  //     FFT[Pr]_k - FFT[Pr]_{k+n/2} = i w_n^k (
  //                                       conj(FFT[Q1]_{-k}) conj(FFT[Q2]_{-k})
  //                                       - FFT[Q1]_k FFT[Q2]_k
  //                                   )
  //
  // For step 5, we thus have
  //     FFT[Qr]_k = (
  //                     (3 - w_{n/2}^k) FFT[Q1]_k FFT[Q2]_k
  //                     + (1 + w_{n/2}^k) FFT[Q1]_k conj(FFT[Q2]_{-k})
  //                     + (1 + w_{n/2}^k) conj(FFT[Q1]_{-k}) FFT[Q2]_k
  //                     - (1 + w_{n/2}^k) conj(FFT[Q1]_{-k}) conj(FFT[Q2]_{-k})
  //                 ) / 4
  // This requires 8 complex multiplications to compute, but can be simplified to
  //     FFT[Qr]_k = FFT[Q1]_k FFT[Q2]_k - (
  //                     (1 + w_{n/2}^k)
  //                     * (FFT[Q1]_k - conj(FFT[Q1]_{-k}))
  //                     * (FFT[Q2]_k - conj(FFT[Q2]_{-k}))
  //                     / 4
  //                 ),
  // which needs only 3 complex multiplications. Why this formula reduced to something this simple
  // is beyond me.
  //
  // Anyway, suppose the functions we actually have are FFT-DIF (decimation in frequency) and
  // FFT-DIT (decimation in time), where
  //     FFT-DIF[P]_k = FFT[P]_{bitreverse(k)}
  //     FFT-DIT[P]_k = FFT[sum_j P_{bitreverse(j)} x^j]_k
  // These functions are both easier to compute than FFT. They also form inverses, in a way:
  //     FFT-DIT[FFT-DIF[P]]_k = FFT[FFT[P]]_k = n P_{-k}
  // The other way to compose them does not work out as nicely:
  //     FFT-DIF[FFT-DIT[P]]_k = n P_{bitreverse(-bitreverse(k))}
  //
  // Nevertheless, it's useful to consider how cache-efficient it is to access P(x) at indices
  // bitreverse(-bitreverse(k)) for k = 0, 1, 2... As -x = ~x + 1, we have
  //     bitreverse(-bitreverse(k)) = bitreverse(bitreverse(~k) + 1),
  // i.e. incrementing ~k at its highest digit in reverse bitorder (as opposed to classic addition).
  // For k = 00...01 || A, we have ~k = 11...10 || ~A, so the formula maps k to 00...01 || ~A. Thus,
  // for each j, the formula reverses the [2^j; 2^{j+1}-1] segment. The sequence is hence:
  //     0; 1; 3, 2; 7, 6, 5, 4; 15, 14, 13, 12, 11, 10, 9, 8; ...
  // This also gives a way to interate through k and obtain the index without any additional
  // computations. As there are O(log n) such segments, there should only be O(log n) cache misses
  // before the predictor aligns to the next segment.
  //
  // To utilize the inverse formula, we can rewrite the steps to do the following:
  // 1. Transform P1(x) and P2(x) to Q1(x) and Q2(x)
  // 2. Compute FFT-DIF[Q1] and FFT-DIF[Q2]
  // 3. Transform FFT-DIF[Q1] and FFT-DIF[Q2] to FFT-DIF[P1] and FFT-DIF[P2]
  // 4. Obtain FFT-DIF[Pr] by pointwise multiplication
  // 5. Transform FFT-DIF[Pr] to FFT-DIF[Qr]
  // 6. Compute FFT-DIT[FFT-DIF[Qr]]_k = n/2 * Qr_{-k}
  // 7. Restore Qr_k from n/2 * Qr_{-k}
  // 8. Transform Qr(x) to Pr(x)
  //
  // Thus steps 3-5 are rewritten like this:
  //     FFT-DIF[Qr]_k = FFT-DIF[Q1]_k FFT-DIF[Q2]_k - (
  //                         (1 + w_{n/2}^{bitreverse(k)})
  //                         * (FFT-DIF[Q1]_k - conj(FFT-DIF[Q1]_{bitreverse(-bitreverse(k))}))
  //                         * (FFT-DIF[Q2]_k - conj(FFT-DIF[Q2]_{bitreverse(-bitreverse(k))}))
  //                         / 4
  //                     )
  // The formula contains the same access pattern as seen before (bitreverse(-bitreverse(k))),
  // which, as we are now aware of, incurs O(log n) cache misses, meaning that all steps but the FFT
  // itself are low-cost.
  //
  // Steps 7-8 can be merged into one step:
  //     Pr_{2k}   = Re Qr_k = Re Qr_{-k} * 2/n
  //     Pr_{2k+1} = Im Qr_k = Im Qr_{-k} * 2/n
  int n_pow = get_fft_n_pow(lhs, rhs);
  ensure_twiddle_factors(n_pow - 1);
  Complex *lhs_fft_dif = mul_fft_transform_input(lhs, n_pow);
  Complex *rhs_fft_dif = mul_fft_transform_input(rhs, n_pow);
  Complex *prod_fft_dif = mul_fft_middle_end(lhs_fft_dif, rhs_fft_dif, n_pow);
  ::operator delete[](lhs_fft_dif, std::align_val_t(32));
  ::operator delete[](rhs_fft_dif, std::align_val_t(32));
  BigInt result = mul_fft_transform_output(prod_fft_dif, n_pow);
  ::operator delete[](prod_fft_dif, std::align_val_t(32));
  ensure(result.data.size() == lhs.data.size() + rhs.data.size() ||
         result.data.size() == lhs.data.size() + rhs.data.size() - 1);
  return result;
}

BigInt::BigInt(SmallVec data) : data(data) {}

BigInt::BigInt() {}

BigInt::BigInt(__uint128_t value) {
  data = {static_cast<uint64_t>(value), static_cast<uint64_t>(value >> 64)};
  data.set_size((value > 0) + (value >= (__uint128_t{1} << 64)));
}
BigInt::BigInt(uint64_t value) {
  data = {value};
  data.set_size(value > 0);
}
BigInt::BigInt(int value) {
  ensure(value >= 0);
  if (value > 0) {
    data = {static_cast<uint64_t>(value)};
  }
}

BigInt::BigInt(const char *s, with_base base)
    : BigInt(std::string_view(s), base) {}

BigInt::BigInt(ConstRef rhs) : data(rhs.data) {}
BigInt::BigInt(const BigInt &rhs) : data(rhs.data) {}
BigInt::BigInt(BigInt &&rhs) : data(std::move(rhs.data)) {}

BigInt &BigInt::operator=(ConstRef rhs) {
  data = rhs.data;
  return *this;
}

BigInt &BigInt::operator=(BigInt &&rhs) {
  data = std::move(rhs.data);
  return *this;
}

bool operator==(ConstRef lhs, ConstRef rhs) { return lhs.data == rhs.data; }
bool operator!=(ConstRef lhs, ConstRef rhs) { return !(lhs == rhs); }

bool operator<(ConstRef lhs, ConstRef rhs) {
  if (lhs.data.size() != rhs.data.size()) {
    return lhs.data.size() < rhs.data.size();
  }
  return std::lexicographical_compare(lhs.data.rbegin(), lhs.data.rend(),
                                      rhs.data.rbegin(), rhs.data.rend());
}
bool operator>(ConstRef lhs, ConstRef rhs) { return rhs < lhs; }
bool operator<=(ConstRef lhs, ConstRef rhs) { return !(rhs < lhs); }
bool operator>=(ConstRef lhs, ConstRef rhs) { return !(lhs < rhs); }

BigInt &BigInt::operator+=(ConstRef rhs) {
  data.increase_size_zerofill(std::max(data.size(), rhs.data.size()) + 1);
  add_to(*this, rhs);
  if (data.back() == 0) {
    data.pop_back();
  }
  return *this;
}

BigInt &BigInt::operator-=(ConstRef rhs) {
  if (rhs.data.empty()) {
    return *this;
  }

  ensure(data.size() >= rhs.data.size());

  size_t offset = 0;
  uint64_t value1, value2;
  size_t unrolled_loop_count = rhs.data.size() / 4;
  size_t left_loop_count = rhs.data.size() % 4;
  if (left_loop_count == 0) {
    unrolled_loop_count--;
    left_loop_count = 4;
  }
  asm volatile(
      "test %[unrolled_loop_count], %[unrolled_loop_count];"
      "jz 2f;"
      "1:"
      "mov (%[rhs_data_ptr],%[offset]), %[value1];"
      "mov 0x8(%[rhs_data_ptr],%[offset]), %[value2];"
      "sbb %[value1], (%[data_ptr],%[offset]);"
      "sbb %[value2], 0x8(%[data_ptr],%[offset]);"
      "mov 0x10(%[rhs_data_ptr],%[offset]), %[value1];"
      "mov 0x18(%[rhs_data_ptr],%[offset]), %[value2];"
      "sbb %[value1], 0x10(%[data_ptr],%[offset]);"
      "sbb %[value2], 0x18(%[data_ptr],%[offset]);"
      "lea 0x20(%[offset]), %[offset];"
      "dec %[unrolled_loop_count];"
      "jnz 1b;"
      "2:"
      "mov (%[rhs_data_ptr],%[offset]), %[value1];"
      "sbb %[value1], (%[data_ptr],%[offset]);"
      "lea 0x8(%[offset]), %[offset];"
      "dec %[left_loop_count];"
      "jnz 2b;"
      "jnc 3f;"
      "4:"
      "subq $1, (%[data_ptr],%[offset]);"
      "lea 0x8(%[offset]), %[offset];"
      "jc 4b;"
      "3:"
      : [offset] "+r"(offset), [value1] "=&r"(value1), [value2] "=&r"(value2),
        [unrolled_loop_count] "+r"(unrolled_loop_count),
        [left_loop_count] "+r"(left_loop_count)
      : [data_ptr] "r"(data.data()), [rhs_data_ptr] "r"(rhs.data.data())
      : "flags", "memory");

  _normalize();
  return *this;
}

BigInt &BigInt::operator+=(const BigInt &rhs) {
  return *this += static_cast<ConstRef>(rhs);
}
BigInt &BigInt::operator-=(const BigInt &rhs) {
  return *this -= static_cast<ConstRef>(rhs);
}

BigInt &BigInt::operator++() { return *this += 1; }
BigInt BigInt::operator++(int) {
  BigInt tmp = *this;
  ++*this;
  return tmp;
}

BigInt &BigInt::operator--() { return *this -= 1; }
BigInt BigInt::operator--(int) {
  BigInt tmp = *this;
  --*this;
  return tmp;
}

BigInt operator+(BigInt lhs, ConstRef rhs) { return lhs += rhs; }
BigInt operator-(BigInt lhs, ConstRef rhs) { return lhs -= rhs; }

BigInt &operator*=(BigInt &lhs, uint64_t rhs) {
  if (rhs == 0) {
    lhs.data.clear_dealloc();
    return lhs;
  }
  uint64_t carry = 0;
  for (size_t i = 0; i < lhs.data.size(); i++) {
    __uint128_t total = __uint128_t{lhs.data[i]} * rhs + carry;
    lhs.data[i] = static_cast<uint64_t>(total);
    carry = static_cast<uint64_t>(total >> 64);
  }
  if (carry != 0) {
    lhs.data.push_back(carry);
  }
  return lhs;
}

uint32_t BigInt::divmod_inplace(uint32_t rhs) {
  uint32_t *lhs_data = reinterpret_cast<uint32_t *>(data.data());
  size_t lhs_size = data.size() * 2;

  uint32_t remainder = 0;
  for (size_t i = lhs_size; i > 0; i--) {
    uint64_t cur = lhs_data[i - 1] | (uint64_t{remainder} << 32);
    lhs_data[i - 1] = static_cast<uint32_t>(cur / rhs);
    remainder = cur % rhs;
  }
  _normalize();
  return remainder;
}

// Credits to https://stackoverflow.com/a/35780435
constexpr uint64_t inv_2_64(uint64_t x) {
  ensure(x % 2 == 1);
  uint64_t r = x;
  for (int i = 0; i < 5; i++) {
    r = r * (2 - r * x);
  }
  ensure(r * x == 1);
  return r;
}

void BigInt::divide_inplace_whole(uint64_t rhs) {
  uint64_t inv = inv_2_64(rhs);
  uint64_t borrow = 0;
  for (size_t i = 0; i < data.size(); i++) {
    uint64_t input_word = data[i];
    uint64_t word = input_word - borrow;
    borrow = word > input_word;
    uint64_t res_word = word * inv;
    data[i] = res_word;
    borrow += __uint128_t{res_word} * rhs >> 64;
  }
  ensure(borrow == 0);
  _normalize();
}

BigInt &operator/=(BigInt &lhs, uint32_t rhs) {
  lhs.divmod_inplace(rhs);
  return lhs;
}

BigInt operator*(BigInt lhs, uint64_t rhs) { return lhs *= rhs; }
BigInt operator/(BigInt lhs, uint32_t rhs) { return lhs /= rhs; }

BigInt operator*(ConstRef lhs, ConstRef rhs);
void mul_to(Ref result, ConstRef lhs, ConstRef rhs);

ConstRef mul_to_ref(Ref result, ConstRef lhs, ConstRef rhs) {
  mul_to(result, lhs, rhs);
  size_t len = 0;
  if (!lhs.data.empty() && !rhs.data.empty()) {
    len = lhs.data.size() + rhs.data.size() - 1;
    if (result.data[len] != 0) {
      len++;
    }
  }
  return result.slice(0, len);
}

ConstRef mul_to_ref_nonzero(Ref result, ConstRef lhs, ConstRef rhs) {
  mul_to(result, lhs, rhs);
  size_t len = lhs.data.size() + rhs.data.size() - 1;
  if (result.data[len] != 0) {
    len++;
  }
  return result.slice(0, len);
}

void mul_1x1(Ref result, ConstRef lhs, ConstRef rhs) {
  __uint128_t product = __uint128_t{lhs.data[0]} * rhs.data[0];
  result.data[0] = static_cast<uint64_t>(product);
  if (product >= (__uint128_t{1} << 64)) {
    result.data[1] = static_cast<uint64_t>(product >> 64);
  }
}

void mul_nx1(Ref result, ConstRef lhs, uint64_t rhs) {
  uint64_t carry = 0;
  for (size_t i = 0; i < lhs.data.size(); i++) {
    __uint128_t total = __uint128_t{lhs.data[i]} * rhs + carry;
    result.data[i] = static_cast<uint64_t>(total);
    carry = static_cast<uint64_t>(total >> 64);
  }
  if (carry != 0) {
    result.data[lhs.data.size()] = carry;
  }
}

__attribute__((noinline)) void mul_quadratic(Ref result, ConstRef lhs,
                                             ConstRef rhs) {
  size_t size = lhs.data.size() + rhs.data.size() - 1;

  uint64_t carry_low = 0;
  uint64_t carry_high = 0;
  for (size_t i = 0; i < size; i++) {
    size_t left = static_cast<size_t>(
        std::max(static_cast<ssize_t>(i + 1 - rhs.data.size()), ssize_t{0}));
    size_t right = std::min(i + 1, lhs.data.size());

    uint64_t sum_low = carry_low;
    uint64_t sum_mid = carry_high;
    uint64_t sum_high = 0;

#define LOOP                                                                   \
  do {                                                                         \
    uint64_t rax = lhs.data[left];                                             \
    asm("mulq %[b];"                                                           \
        "add %%rax, %[sum_low];"                                               \
        "adc %%rdx, %[sum_mid];"                                               \
        "adc $0, %[sum_high];"                                                 \
        : "+a"(rax), [sum_low] "+r"(sum_low), [sum_mid] "+r"(sum_mid),         \
          [sum_high] "+r"(sum_high)                                            \
        : [b] "m"(rhs.data[i - left])                                          \
        : "flags", "rdx");                                                     \
    left++;                                                                    \
  } while (0)

    while (left + 8 <= right) {
      LOOP;
      LOOP;
      LOOP;
      LOOP;
      LOOP;
      LOOP;
      LOOP;
      LOOP;
    }
    while (left < right) {
      LOOP;
    }

    result.data[i] = sum_low;
    carry_low = sum_mid;
    carry_high = sum_high;
  }

  if (carry_high > 0) {
    result.data[size] = carry_low;
    result.data[size + 1] = carry_high;
  } else if (carry_low > 0) {
    result.data[size] = carry_low;
  }
}

void mul_karatsuba(Ref result, ConstRef lhs, ConstRef rhs) {
  size_t b = std::min(lhs.data.size(), rhs.data.size()) / 2;

  ConstRef x0 = lhs.slice(0, b).normalized();
  ConstRef x1 = lhs.slice(b);
  ConstRef y0 = rhs.slice(0, b).normalized();
  ConstRef y1 = rhs.slice(b);

  ConstRef z0 = mul_to_ref(result, x0, y0);
  ConstRef z2 = mul_to_ref_nonzero(result.slice(b * 2), x1, y1);

  add_to(result.slice(b), (x0 + x1) * (y0 + y1) - z0 - z2);
}

void mul_toom33(Ref result, ConstRef lhs, ConstRef rhs) {
  // Split lhs and rhs into
  //   lhs: p(x) = a0 + a1 x + a2 x^2
  //   rhs: q(x) = b0 + b1 x + b2 x^2
  // choosing a size such that we necessarily have a2 > a1, b2 > b1
  size_t b = std::min(lhs.data.size(), rhs.data.size()) / 3 - 1;
  ensure(b * 2 <= lhs.data.size());
  ensure(b * 2 <= rhs.data.size());

  ConstRef a0 = lhs.slice(0, b).normalized();
  ConstRef a1 = lhs.slice(b, b).normalized();
  ConstRef a2 = lhs.slice(b * 2);
  ConstRef b0 = rhs.slice(0, b).normalized();
  ConstRef b1 = rhs.slice(b, b).normalized();
  ConstRef b2 = rhs.slice(b * 2);

  // The algorithm works as follows
  // We compute r(x) = p(x) q(x) at points
  //   r(0) = a0 b0
  //   r(1) = (a0 + a1 + a2) (b0 + b1 + b2)
  //   r(-1) = (a0 - a1 + a2) (b0 - b1 + b2)
  //   r(2) = (a0 + 2 a1 + 4 a2) (b0 + 2 b1 + 4 b2)
  //   r(inf) = a2 b2
  // We then substitute r(x) = c0 + c1 x + c2 x^2 + c3 x^3 + c4 x^4
  //   ( r(0) ) = (1  0  0  0  0) (с0)
  //   ( r(1) ) = (1  1  1  1  1) (с1)
  //   (r(-1) ) = (1 -1  1 -1  1) (с2)
  //   ( r(2) ) = (1  2  4  8 16) (с3)
  //   (r(inf)) = (0  0  0  0  1) (с4)
  // Therefore
  //   (c0)   ( 1    0    0    0    0) ( r(0) )
  //   (c1)   (-1/2  1   -1/3 -1/6  2) ( r(1) )
  //   (c2) = (-1    1/2  1/2  0   -1) (r(-1) )
  //   (c3)   ( 1/2 -1/2 -1/6  1/6 -2) ( r(2) )
  //   (c4)   ( 0    0    0    0    1) (r(inf))

  // We compute c0 = r(0) and c4 = r(inf) directly into their respective
  // locations
  ConstRef c0 = mul_to_ref(result, a0, b0);
  ConstRef c4 = mul_to_ref(result.slice(b * 4), a2, b2);
  ConstRef r0 = c0;
  ConstRef rinf = c4;

  // Compute r at other points
  BigInt a0_plus_a2 = a0 + a2;
  BigInt b0_plus_b2 = b0 + b2;
  BigInt a0_plus_a1_plus_a2 = a0_plus_a2 + a1;
  BigInt b0_plus_b1_plus_b2 = b0_plus_b2 + b1;
  BigInt r1 = a0_plus_a1_plus_a2 * b0_plus_b1_plus_b2;
  BigInt rm1 = (std::move(a0_plus_a2) - a1) * (std::move(b0_plus_b2) - b1);
  BigInt r2 = (std::move(a0_plus_a1_plus_a2) + a1 + a2 * 3) *
              (std::move(b0_plus_b1_plus_b2) + b1 + b2 * 3);
  BigInt rinf_2 = rinf * 2;

  auto half = [](BigInt a) {
    a.halve();
    return a;
  };
  auto third = [](BigInt a) {
    a.divide_inplace_whole(3);
    return a;
  };

  // Compute other c's in separate memory because they would otherwise override
  // c4
  BigInt a = third(half(r0 * 3 + r2) + rm1);
  BigInt c3 = half(third(std::move(r2) - rm1) + r0 - r1) - rinf_2;
  BigInt c1 = std::move(rinf_2) + r1 - a;
  BigInt c2 = half(std::move(r1) + rm1) - rinf - r0;

  add_to(result.slice(b), c1);
  add_to(result.slice(b * 2), c2);
  add_to(result.slice(b * 3), c3);
}

void mul_disproportional(Ref result, ConstRef lhs, ConstRef rhs) {
  ensure(lhs.data.size() < rhs.data.size());
  mul_to(result, lhs, rhs.slice(0, lhs.data.size()).normalized());
  size_t i = lhs.data.size();
  for (; i + lhs.data.size() < rhs.data.size(); i += lhs.data.size()) {
    add_to(result.slice(i), lhs * rhs.slice(i, lhs.data.size()).normalized());
  }
  add_to(result.slice(i), lhs * rhs.slice(i));
}

void mul_to(Ref result, ConstRef lhs, ConstRef rhs) {
  if (lhs.data.empty() || rhs.data.empty()) {
    return;
  }

  if (lhs.data.size() == 1 && rhs.data.size() == 1) {
    mul_1x1(result, lhs, rhs);
  } else if (rhs.data.size() == 1) {
    mul_nx1(result, lhs, rhs.data[0]);
  } else if (lhs.data.size() == 1) {
    mul_nx1(result, rhs, lhs.data[0]);
  } else if (std::min(lhs.data.size(), rhs.data.size()) >= 40) {
    if (lhs.data.size() * 2 < rhs.data.size()) {
      mul_disproportional(result, lhs, rhs);
    } else if (rhs.data.size() * 2 < lhs.data.size()) {
      mul_disproportional(result, rhs, lhs);
    } else if (std::min(lhs.data.size(), rhs.data.size()) >= 2000) {
      mul_toom33(result, lhs, rhs);
    } else {
      mul_karatsuba(result, lhs, rhs);
    }
  } else {
    mul_quadratic(result, lhs, rhs);
  }
}

BigInt operator*(ConstRef lhs, ConstRef rhs) {
  if (lhs.data.empty() || rhs.data.empty()) {
    return {};
  }
  int n_pow = get_fft_n_pow(lhs, rhs);
  if (n_pow >= FFT_CUTOFF && n_pow <= FFT_MAX) {
    return mul_fft(lhs, rhs);
  }
  BigInt result;
  result.data.increase_size_zerofill(lhs.data.size() + rhs.data.size());
  mul_to(result, lhs, rhs);
  result._normalize_nonzero();
  return result;
}

template <typename Iterator, typename Map>
uint64_t str_to_int_64(Iterator begin, Iterator end, uint64_t base, Map map) {
  uint64_t val = 0;
  for (auto it = end; it != begin;) {
    val *= base;
    val += map(*--it);
  }
  return val;
}

template <typename Iterator, typename Map>
__uint128_t str_to_int_128(Iterator begin, Iterator end, uint64_t base,
                           int max_block_len, uint64_t base_product, Map map) {
  uint64_t low = str_to_int_64(begin, begin + max_block_len, base, map);
  uint64_t high = str_to_int_64(begin + max_block_len, end, base, map);
  return static_cast<__uint128_t>(high) * base_product + low;
}

template <typename Iterator, typename Map>
void str_to_int_inplace(Iterator begin, Iterator end, uint64_t base, Map map,
                        const BigInt *powers_of_base, int max_block_len,
                        uint64_t base_product, BigInt &result) {
  if (end - begin <= max_block_len) {
    result += str_to_int_64(begin, end, base, map);
    return;
  } else if (end - begin <= 2 * max_block_len) {
    result +=
        str_to_int_128(begin, end, base, max_block_len, base_product, map);
    return;
  } else if (end - begin <= 200 * max_block_len) {
    int first_block_len = static_cast<int>((end - begin) % max_block_len);
    if (first_block_len == 0) {
      first_block_len = max_block_len;
    }
    BigInt tmp = str_to_int_64(end - first_block_len, end, base, map);
    for (end -= first_block_len; begin != end; end -= max_block_len) {
      tmp *= base_product;
      tmp += str_to_int_64(end - max_block_len, end, base, map);
    }
    result += tmp;
    return;
  }

  int low_len_pow =
      63 - __builtin_clzll(static_cast<uint64_t>(end - begin - 1));
  ssize_t low_len = ssize_t{1} << low_len_pow;
  Iterator mid = begin + low_len;
  BigInt high;
  str_to_int_inplace(mid, end, base, map, powers_of_base, max_block_len,
                     base_product, high);
  result += high * powers_of_base[low_len_pow];
  str_to_int_inplace(begin, mid, base, map, powers_of_base, max_block_len,
                     base_product, result);
}

template <typename Iterator, typename Map>
BigInt str_to_int(Iterator begin, Iterator end, uint64_t base, Map map) {
  int max_block_len = 0;
  uint64_t base_product = 1;
  while (base_product <= static_cast<uint64_t>(-1) / base) {
    max_block_len++;
    base_product *= base;
  }

  std::vector<BigInt> powers_of_base{base};
  while ((ssize_t{1} << powers_of_base.size()) <= end - begin) {
    powers_of_base.push_back(powers_of_base.back() * powers_of_base.back());
  }

  BigInt result;
  str_to_int_inplace(begin, end, base, map, powers_of_base.data(),
                     max_block_len, base_product, result);
  return result;
}

template <typename List, typename> BigInt::BigInt(List &&list, with_base base) {
  *this = str_to_int(list.begin(), list.end(), base.base,
                     [](uint64_t digit) { return digit; });
}
BigInt::BigInt(std::string_view s, with_base base) {
  ensure(base.base <= 36);
  *this = str_to_int(s.rbegin(), s.rend(), base.base, [base](char c) {
    uint64_t digit;
    if ('0' <= c && c <= '9') {
      digit = static_cast<uint64_t>(c - '0');
    } else if ('a' <= c && c <= 'z') {
      digit = static_cast<uint64_t>(c - 'a' + 10);
    } else {
      ensure(false);
    }
    ensure(digit < base.base);
    return digit;
  });
}

std::ostream &operator<<(std::ostream &out, ConstRef rhs) {
  if (rhs.data.empty()) {
    return out << "0x0";
  }
  out << "0x" << std::hex << rhs.data.back() << std::setfill('0');
  for (auto it = rhs.data.rbegin() + 1; it != rhs.data.rend(); ++it) {
    out << std::setw(16) << *it;
  }
  return out << std::dec << std::setfill(' ');
}

} // namespace EXPORT bigint

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

void memzero64(uint64_t* data, size_t count) {
  asm volatile("rep stosq" : "+D"(data), "+c"(count) : "a"(0) : "memory");
}

void memcpy64(uint64_t* dst, const uint64_t* src, size_t count) {
  asm volatile("rep movsq" : "+D"(dst), "+S"(src), "+c"(count) : : "memory");
}

struct EXPORT with_base {
  uint64_t base;
};

class ConstSpan;

class SmallVec {
  static constexpr size_t INLINE_STORAGE_SIZE = 8;

  uint64_t* _begin;
  size_t _size;
  size_t _capacity;
  uint64_t _inline_storage[INLINE_STORAGE_SIZE];

  void increase_capacity_to(size_t new_capacity) {
    uint64_t* new_begin = new uint64_t[new_capacity];
    memcpy64(new_begin, _begin, _size);
    if (_begin != _inline_storage) {
      delete[] _begin;
    }
    _begin = new_begin;
    _capacity = new_capacity;
  }

public:
  SmallVec() : _begin(_inline_storage), _size(0), _capacity(INLINE_STORAGE_SIZE) {}
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
  SmallVec(const uint64_t* data, size_t size) {
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
  SmallVec(const SmallVec& rhs);
  SmallVec(SmallVec&& rhs) {
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

  SmallVec& operator=(ConstSpan rhs);
  SmallVec& operator=(SmallVec&& rhs) {
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

  uint64_t& operator[](size_t i) { return _begin[i]; }
  const uint64_t& operator[](size_t i) const { return _begin[i]; }

  size_t size() const { return _size; }
  bool empty() const { return _size == 0; }
  uint64_t* data() { return _begin; }
  const uint64_t* data() const { return _begin; }

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

  bool operator==(const SmallVec& rhs) const {
    return _size == rhs._size && std::equal(_begin, _begin + _size, rhs._begin);
  }

  uint64_t& back() { return _begin[_size - 1]; }
  const uint64_t& back() const { return _begin[_size - 1]; }

  uint64_t* begin() { return _begin; }
  uint64_t* end() { return _begin + _size; }
  const uint64_t* begin() const { return _begin; }
  const uint64_t* end() const { return _begin + _size; }

  std::reverse_iterator<uint64_t*> rbegin() { return std::make_reverse_iterator(_begin + _size); }
  std::reverse_iterator<uint64_t*> rend() { return std::make_reverse_iterator(_begin); }
  std::reverse_iterator<const uint64_t*> rbegin() const {
    return std::make_reverse_iterator(_begin + _size);
  }
  std::reverse_iterator<const uint64_t*> rend() const { return std::make_reverse_iterator(_begin); }
};

class Span {
  uint64_t* _begin;
  size_t _size;

public:
  Span() : _begin(nullptr), _size(0) {}
  Span(uint64_t* data, size_t size) : _begin(data), _size(size) {}
  Span(SmallVec& vec) : _begin(vec.begin()), _size(vec.size()) {}
  void set_size(size_t size) { _size = size; }

  uint64_t& operator[](size_t i) { return _begin[i]; }
  const uint64_t& operator[](size_t i) const { return _begin[i]; }

  size_t size() const { return _size; }
  bool empty() const { return _size == 0; }
  uint64_t* data() { return _begin; }
  const uint64_t* data() const { return _begin; }

  void pop_back() { _size--; }

  bool operator==(const Span& rhs) const {
    return _size == rhs._size && std::equal(_begin, _begin + _size, rhs._begin);
  }

  uint64_t& back() { return _begin[_size - 1]; }
  uint64_t back() const { return _begin[_size - 1]; }

  uint64_t* begin() { return _begin; }
  uint64_t* end() { return _begin + _size; }
  const uint64_t* begin() const { return _begin; }
  const uint64_t* end() const { return _begin + _size; }

  std::reverse_iterator<uint64_t*> rbegin() { return std::make_reverse_iterator(_begin + _size); }
  std::reverse_iterator<uint64_t*> rend() { return std::make_reverse_iterator(_begin); }
  std::reverse_iterator<const uint64_t*> rbegin() const {
    return std::make_reverse_iterator(_begin + _size);
  }
  std::reverse_iterator<const uint64_t*> rend() const { return std::make_reverse_iterator(_begin); }
};

class ConstSpan {
  const uint64_t* _begin;
  size_t _size;

public:
  ConstSpan() : _begin(nullptr), _size(0) {}
  ConstSpan(const uint64_t* data, size_t size) : _begin(data), _size(size) {}
  ConstSpan(const SmallVec& vec) : _begin(vec.begin()), _size(vec.size()) {}
  ConstSpan(Span span) : _begin(span.begin()), _size(span.size()) {}
  void set_size(size_t size) { _size = size; }

  const uint64_t& operator[](size_t i) const { return _begin[i]; }

  size_t size() const { return _size; }
  bool empty() const { return _size == 0; }
  const uint64_t* data() const { return _begin; }

  void pop_back() { _size--; }

  bool operator==(const ConstSpan& rhs) const {
    return _size == rhs._size && std::equal(_begin, _begin + _size, rhs._begin);
  }

  const uint64_t& back() const { return _begin[_size - 1]; }

  const uint64_t* begin() const { return _begin; }
  const uint64_t* end() const { return _begin + _size; }

  std::reverse_iterator<const uint64_t*> rbegin() const {
    return std::make_reverse_iterator(_begin + _size);
  }
  std::reverse_iterator<const uint64_t*> rend() const { return std::make_reverse_iterator(_begin); }
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

SmallVec::SmallVec(const SmallVec& rhs) : SmallVec(static_cast<ConstSpan>(rhs)) {}

SmallVec& SmallVec::operator=(ConstSpan rhs) {
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
  BigInt(List&& list, with_base base);
  BigInt(std::string_view s, with_base base = {10});
  BigInt(const char* s, with_base base = {10});

  BigInt(ConstRef rhs);
  BigInt(const BigInt& rhs);
  BigInt(BigInt&& rhs);

  BigInt& operator=(ConstRef rhs);
  BigInt& operator=(BigInt&& rhs);

  BigInt& operator+=(ConstRef rhs);
  BigInt& operator-=(ConstRef rhs);
  BigInt& operator+=(const BigInt& rhs);
  BigInt& operator-=(const BigInt& rhs);

  BigInt& operator++();
  BigInt operator++(int);

  BigInt& operator--();
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
  Ref(BigInt& bigint) : data(bigint.data) {}
  Ref(BigInt&& bigint) : data(bigint.data) {}

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
  ConstRef(const BigInt& bigint) : data(bigint.data) {}
  ConstRef(Ref ref) : data(ref.data) {}

  ConstRef slice(size_t l) const { return {ConstSpan{data.data() + l, data.size() - l}}; }
  ConstRef slice(size_t l, size_t size) const { return {ConstSpan{data.data() + l, size}}; }

  explicit operator BigInt() const { return {SmallVec{data.data(), data.size()}}; }

  ConstRef normalized() {
    ConstSpan tmp = data;
    while (!tmp.empty() && tmp.back() == 0) {
      tmp.pop_back();
    }
    return {tmp};
  }
};

ConstRef Ref::slice(size_t l) const { return static_cast<ConstRef>(*this).slice(l); }
ConstRef Ref::slice(size_t l, size_t size) const {
  return static_cast<ConstRef>(*this).slice(l, size);
}

Ref BigInt::slice(size_t l) { return static_cast<Ref>(*this).slice(l); }
Ref BigInt::slice(size_t l, size_t size) { return static_cast<Ref>(*this).slice(l, size); }
ConstRef BigInt::slice(size_t l) const { return static_cast<ConstRef>(*this).slice(l); }
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
        [unrolled_loop_count] "+r"(unrolled_loop_count), [left_loop_count] "+r"(left_loop_count)
      : [data_ptr] "r"(lhs.data.data()), [rhs_data_ptr] "r"(rhs.data.data())
      : "flags", "memory");
}

inline constexpr int FFT_CUTOFF = 10;
inline constexpr int FFT_RECURSIVE = 10;
inline constexpr int FFT_MAX_16BIT = 19;  // a bit less than 52 - 16 * 2
inline constexpr int FFT_MAX_12BIT = 26;  // XXX: this bound is unverified

double* construct_initial_twiddles_bitreversed() {
  // bitreversed k | length | k | n | 2 pi k / n | cos      | sin
  // 0             | 0      | 0 | 2 | 0          | 1        | 0
  // 1             | 1      | 1 | 4 | pi/2       | 0        | 1
  return new (std::align_val_t(32)) double[4]{1, 0, 0, 1};
}

inline double* twiddles_bitreversed = construct_initial_twiddles_bitreversed();
inline int twiddles_n_pow = 2;

uint32_t bitreverse(uint32_t k, int n_pow) {
  k <<= 32 - n_pow;
  k = ((k & 0x55555555) << 1) | ((k >> 1) & 0x55555555);
  k = ((k & 0x33333333) << 2) | ((k >> 2) & 0x33333333);
  k = ((k & 0x0f0f0f0f) << 4) | ((k >> 4) & 0x0f0f0f0f);
  k = ((k & 0x00ff00ff) << 8) | ((k >> 8) & 0x00ff00ff);
  k = ((k & 0x0000ffff) << 16) | ((k >> 16) & 0x0000ffff);
  return k;
}

void ensure_twiddle_factors(int want_n_pow) {
  if (twiddles_n_pow >= want_n_pow) {
    return;
  }

  static constexpr double PI = 3.1415926535897931;

  // twiddle_bitreversed doesn't depend on n (to be more specific, twiddle_bitreversed of a lesser n
  // is a prefix of twiddle_bitreversed of a larger n)
  size_t old_n = 1uz << twiddles_n_pow;
  size_t new_n = 1uz << want_n_pow;
  auto new_twiddles_bitreversed = new (std::align_val_t(32)) double[new_n];
  memcpy64(reinterpret_cast<uint64_t*>(new_twiddles_bitreversed),
           reinterpret_cast<uint64_t*>(twiddles_bitreversed), old_n / 2);
  memcpy64(reinterpret_cast<uint64_t*>(new_twiddles_bitreversed) + new_n / 2,
           reinterpret_cast<uint64_t*>(twiddles_bitreversed + old_n / 2), old_n / 2);
  ::operator delete[](twiddles_bitreversed, std::align_val_t(32));
  twiddles_bitreversed = new_twiddles_bitreversed;
  double coeff = 2 * PI / static_cast<double>(new_n);
  for (size_t k = old_n / 4; k < new_n / 4; k++) {
    double angle = coeff * bitreverse(k, want_n_pow - 2);
    double c = std::cos(angle);
    double s = std::sin(angle);
    twiddles_bitreversed[2 * k] = c;
    twiddles_bitreversed[2 * k + 1] = -s;
    twiddles_bitreversed[2 * k + new_n / 2] = s;
    twiddles_bitreversed[2 * k + 1 + new_n / 2] = c;
  }

  twiddles_n_pow = want_n_pow;
}

int get_fft_n_pow_16bit(ConstRef lhs, ConstRef rhs) {
  // Input is split into 16-bit words, pairs of which are then merged to complex numbers. In total,
  // n 64-bit words are transformed into 2n complex numbers
  return 64 - __builtin_clzll((lhs.data.size() + rhs.data.size()) * 2 - 1);
}

int get_fft_n_pow_12bit(ConstRef lhs, ConstRef rhs) {
  // Input is split into 12-bit words, pairs of which are then merged to complex numbers. In total,
  // 3n 64-bit words are transformed into 8n complex numbers
  return 64 - __builtin_clzll((lhs.data.size() + rhs.data.size()) * 8 / 3 + 1);
}

std::array<__m256d, 4> transpose(__m256d a0, __m256d a1, __m256d a2, __m256d a3) {
  __m256d b0 = _mm256_unpacklo_pd(a0, a1);
  __m256d b1 = _mm256_unpackhi_pd(a0, a1);
  __m256d b2 = _mm256_unpacklo_pd(a2, a3);
  __m256d b3 = _mm256_unpackhi_pd(a2, a3);
  return {_mm256_permute2f128_pd(b0, b2, 0x20), _mm256_permute2f128_pd(b1, b3, 0x20),
          _mm256_permute2f128_pd(b0, b2, 0x31), _mm256_permute2f128_pd(b1, b3, 0x31)};
}

void fft_dif(double* real, double* imag, int n_pow, size_t k_base) {
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
  // where bitreverse is of length log_2 n - 1.
  //
  // FFT-DIF of an array A can thus be computed recursively as follows:
  //     def FFT-DIF(A):
  //         FFT-DIF(A[::2])
  //         FFT-DIF(A[1::2])
  //         for k in range(n/2):
  //             e, o, w = A[2k], A[2k+1], W_bitreversed[k]
  //             A[2k] = e + w * o
  //             A[2k+1] = e - w * o
  // Or iteratively as follows:
  //     def FFT-DIF(A):
  //         for step in range(log_2 n - 1, -1, -1):
  //             for k in range(n / 2^{step+1}):
  //                 for j in range(2k * 2^step, (2k+1) * 2^step):
  //                     e, o, w = A[j], A[j+2^step], W_bitreversed[k]
  //                     A[j] = e + w * o
  //                     A[j+2^step] = e - w * o

  // Non-vectorized code:
  //     for (int step = n_pow - 1; step >= 0; step--) {
  //       for (size_t k = 0; k < (1uz << (n_pow - 1 - step)); k++) {
  //         double w_real = twiddles_bitreversed[k];
  //         double w_imag = twiddles_bitreversed[k + (1uz << (twiddles_n_pow - 1))];
  //         for (size_t j = (2 * k) << step; j < ((2 * k + 1) << step); j++) {
  //           double e_real = real[j];
  //           double e_imag = imag[j];
  //           double o_real = real[j + (1uz << step)];
  //           double o_imag = imag[j + (1uz << step)];
  //           double wo_real = w_real * o_real - w_imag * o_imag;
  //           double wo_imag = w_real * o_imag + w_imag * o_real;
  //           real[j] = e_real + wo_real;
  //           imag[j] = e_imag + wo_imag;
  //           real[j + (1uz << step)] = e_real - wo_real;
  //           imag[j + (1uz << step)] = e_imag - wo_imag;
  //         }
  //       }
  //     }

  int step = n_pow - 1;

  // It's critical for performance to perform multiple steps at once
  auto radix4_high = [&]() {
    for (size_t k = k_base; k < k_base + (1uz << (n_pow - 1 - step)); k++) {
      __m256d w_real = _mm256_set1_pd(twiddles_bitreversed[k]);
      __m256d w_imag = _mm256_set1_pd(twiddles_bitreversed[k + (1uz << (twiddles_n_pow - 1))]);
      __m256d w2_real = _mm256_set1_pd(twiddles_bitreversed[2 * k]);
      __m256d w2_imag = _mm256_set1_pd(twiddles_bitreversed[2 * k + (1uz << (twiddles_n_pow - 1))]);
      __m256d w3_real = _mm256_set1_pd(twiddles_bitreversed[2 * k + 1]);
      __m256d w3_imag =
          _mm256_set1_pd(twiddles_bitreversed[2 * k + 1 + (1uz << (twiddles_n_pow - 1))]);

      for (size_t j = (4 * k) << (step - 1); j < ((4 * k + 1) << (step - 1)); j += 4) {
        __m256d a0_real = _mm256_load_pd(&real[j]);
        __m256d a0_imag = _mm256_load_pd(&imag[j]);
        __m256d a1_real = _mm256_load_pd(&real[j + (1uz << (step - 1))]);
        __m256d a1_imag = _mm256_load_pd(&imag[j + (1uz << (step - 1))]);
        __m256d a2_real = _mm256_load_pd(&real[j + (2uz << (step - 1))]);
        __m256d a2_imag = _mm256_load_pd(&imag[j + (2uz << (step - 1))]);
        __m256d a3_real = _mm256_load_pd(&real[j + (3uz << (step - 1))]);
        __m256d a3_imag = _mm256_load_pd(&imag[j + (3uz << (step - 1))]);

        __m256d wa2_real = _mm256_fmsub_pd(w_real, a2_real, _mm256_mul_pd(w_imag, a2_imag));
        __m256d wa2_imag = _mm256_fmadd_pd(w_real, a2_imag, _mm256_mul_pd(w_imag, a2_real));
        __m256d wa3_real = _mm256_fmsub_pd(w_real, a3_real, _mm256_mul_pd(w_imag, a3_imag));
        __m256d wa3_imag = _mm256_fmadd_pd(w_real, a3_imag, _mm256_mul_pd(w_imag, a3_real));

        __m256d e2_real = _mm256_add_pd(a0_real, wa2_real);
        __m256d e2_imag = _mm256_add_pd(a0_imag, wa2_imag);
        __m256d o2_real = _mm256_add_pd(a1_real, wa3_real);
        __m256d o2_imag = _mm256_add_pd(a1_imag, wa3_imag);
        __m256d e3_real = _mm256_sub_pd(a0_real, wa2_real);
        __m256d e3_imag = _mm256_sub_pd(a0_imag, wa2_imag);
        __m256d o3_real = _mm256_sub_pd(a1_real, wa3_real);
        __m256d o3_imag = _mm256_sub_pd(a1_imag, wa3_imag);

        __m256d wo2_real = _mm256_fmsub_pd(w2_real, o2_real, _mm256_mul_pd(w2_imag, o2_imag));
        __m256d wo2_imag = _mm256_fmadd_pd(w2_real, o2_imag, _mm256_mul_pd(w2_imag, o2_real));
        __m256d wo3_real = _mm256_fmsub_pd(w3_real, o3_real, _mm256_mul_pd(w3_imag, o3_imag));
        __m256d wo3_imag = _mm256_fmadd_pd(w3_real, o3_imag, _mm256_mul_pd(w3_imag, o3_real));

        _mm256_store_pd(&real[j], _mm256_add_pd(e2_real, wo2_real));
        _mm256_store_pd(&imag[j], _mm256_add_pd(e2_imag, wo2_imag));
        _mm256_store_pd(&real[j + (1uz << (step - 1))], _mm256_sub_pd(e2_real, wo2_real));
        _mm256_store_pd(&imag[j + (1uz << (step - 1))],
                        _mm256_sub_pd(e2_imag, wo2_imag));
        _mm256_store_pd(&real[j + (2uz << (step - 1))], _mm256_add_pd(e3_real, wo3_real));
        _mm256_store_pd(&imag[j + (2uz << (step - 1))],
                        _mm256_add_pd(e3_imag, wo3_imag));
        _mm256_store_pd(&real[j + (3uz << (step - 1))], _mm256_sub_pd(e3_real, wo3_real));
        _mm256_store_pd(&imag[j + (3uz << (step - 1))],
                        _mm256_sub_pd(e3_imag, wo3_imag));
      }
    }
  };

  auto radix4_low = [&]() {
    ensure(step == 1);

    for (size_t k = k_base; k < k_base + (1uz << (n_pow - 2)); k += 4) {
      __m256d w_real = _mm256_load_pd(&twiddles_bitreversed[k]);
      __m256d w_imag = _mm256_load_pd(&twiddles_bitreversed[k + (1uz << (twiddles_n_pow - 1))]);

      __m256d w23_real_low = _mm256_load_pd(&twiddles_bitreversed[2 * k]);
      __m256d w23_real_high = _mm256_load_pd(&twiddles_bitreversed[2 * k + 4]);
      __m256d w2_real =
          _mm256_permute4x64_pd(_mm256_unpacklo_pd(w23_real_low, w23_real_high), 0b11011000);
      __m256d w3_real =
          _mm256_permute4x64_pd(_mm256_unpackhi_pd(w23_real_low, w23_real_high), 0b11011000);

      __m256d w23_imag_low =
          _mm256_load_pd(&twiddles_bitreversed[2 * k + (1uz << (twiddles_n_pow - 1))]);
      __m256d w23_imag_high =
          _mm256_load_pd(&twiddles_bitreversed[2 * k + 4 + (1uz << (twiddles_n_pow - 1))]);
      __m256d w2_imag =
          _mm256_permute4x64_pd(_mm256_unpacklo_pd(w23_imag_low, w23_imag_high), 0b11011000);
      __m256d w3_imag =
          _mm256_permute4x64_pd(_mm256_unpackhi_pd(w23_imag_low, w23_imag_high), 0b11011000);

      auto [a0_real, a1_real, a2_real, a3_real] =
          transpose(_mm256_load_pd(&real[4 * k]), _mm256_load_pd(&real[4 * (k + 1)]),
                    _mm256_load_pd(&real[4 * (k + 2)]), _mm256_load_pd(&real[4 * (k + 3)]));
      auto [a0_imag, a1_imag, a2_imag, a3_imag] = transpose(
          _mm256_load_pd(&imag[4 * k]), _mm256_load_pd(&imag[4 * (k + 1)]),
          _mm256_load_pd(&imag[4 * (k + 2)]),
          _mm256_load_pd(&imag[4 * (k + 3)]));

      __m256d wa2_real = _mm256_fmsub_pd(w_real, a2_real, _mm256_mul_pd(w_imag, a2_imag));
      __m256d wa2_imag = _mm256_fmadd_pd(w_real, a2_imag, _mm256_mul_pd(w_imag, a2_real));
      __m256d wa3_real = _mm256_fmsub_pd(w_real, a3_real, _mm256_mul_pd(w_imag, a3_imag));
      __m256d wa3_imag = _mm256_fmadd_pd(w_real, a3_imag, _mm256_mul_pd(w_imag, a3_real));

      __m256d e2_real = _mm256_add_pd(a0_real, wa2_real);
      __m256d e2_imag = _mm256_add_pd(a0_imag, wa2_imag);
      __m256d o2_real = _mm256_add_pd(a1_real, wa3_real);
      __m256d o2_imag = _mm256_add_pd(a1_imag, wa3_imag);
      __m256d e3_real = _mm256_sub_pd(a0_real, wa2_real);
      __m256d e3_imag = _mm256_sub_pd(a0_imag, wa2_imag);
      __m256d o3_real = _mm256_sub_pd(a1_real, wa3_real);
      __m256d o3_imag = _mm256_sub_pd(a1_imag, wa3_imag);

      __m256d wo2_real = _mm256_fmsub_pd(w2_real, o2_real, _mm256_mul_pd(w2_imag, o2_imag));
      __m256d wo2_imag = _mm256_fmadd_pd(w2_real, o2_imag, _mm256_mul_pd(w2_imag, o2_real));
      __m256d wo3_real = _mm256_fmsub_pd(w3_real, o3_real, _mm256_mul_pd(w3_imag, o3_imag));
      __m256d wo3_imag = _mm256_fmadd_pd(w3_real, o3_imag, _mm256_mul_pd(w3_imag, o3_real));

      auto [b0_real, b1_real, b2_real, b3_real] =
          transpose(_mm256_add_pd(e2_real, wo2_real), _mm256_sub_pd(e2_real, wo2_real),
                    _mm256_add_pd(e3_real, wo3_real), _mm256_sub_pd(e3_real, wo3_real));
      _mm256_store_pd(&real[4 * k], b0_real);
      _mm256_store_pd(&real[4 * (k + 1)], b1_real);
      _mm256_store_pd(&real[4 * (k + 2)], b2_real);
      _mm256_store_pd(&real[4 * (k + 3)], b3_real);

      auto [b0_imag, b1_imag, b2_imag, b3_imag] =
          transpose(_mm256_add_pd(e2_imag, wo2_imag), _mm256_sub_pd(e2_imag, wo2_imag),
                    _mm256_add_pd(e3_imag, wo3_imag), _mm256_sub_pd(e3_imag, wo3_imag));
      _mm256_store_pd(&imag[4 * k], b0_imag);
      _mm256_store_pd(&imag[4 * (k + 1)], b1_imag);
      _mm256_store_pd(&imag[4 * (k + 2)], b2_imag);
      _mm256_store_pd(&imag[4 * (k + 3)], b3_imag);
    }
  };

  auto radix2_high = [&]() {
    for (size_t k = k_base; k < k_base + (1uz << (n_pow - 1 - step)); k++) {
      __m256d w_real = _mm256_set1_pd(twiddles_bitreversed[k]);
      __m256d w_imag = _mm256_set1_pd(twiddles_bitreversed[k + (1uz << (twiddles_n_pow - 1))]);
      for (size_t j = (2 * k) << step; j < ((2 * k + 1) << step); j += 4) {
        __m256d e_real = _mm256_load_pd(&real[j]);
        __m256d e_imag = _mm256_load_pd(&imag[j]);
        __m256d o_real = _mm256_load_pd(&real[j + (1uz << step)]);
        __m256d o_imag = _mm256_load_pd(&imag[j + (1uz << step)]);
        __m256d wo_real = _mm256_fmsub_pd(w_real, o_real, _mm256_mul_pd(w_imag, o_imag));
        __m256d wo_imag = _mm256_fmadd_pd(w_real, o_imag, _mm256_mul_pd(w_imag, o_real));
        _mm256_store_pd(&real[j], _mm256_add_pd(e_real, wo_real));
        _mm256_store_pd(&imag[j], _mm256_add_pd(e_imag, wo_imag));
        _mm256_store_pd(&real[j + (1uz << step)], _mm256_sub_pd(e_real, wo_real));
        _mm256_store_pd(&imag[j + (1uz << step)], _mm256_sub_pd(e_imag, wo_imag));
      }
    }
  };

  // If n_pow >= FFT_RECURSIVE, we want to do one radix-4 step and recurse. Otherwise, apply various
  // steps iteratively. We want to only have one invocation of radix4_high so that it is inlined, so
  // the code is a bit more messy than it could be.

  if (n_pow < FFT_RECURSIVE) {
    // Get to odd step by radix-2
    if (step % 2 == 0) {
      radix2_high();
      k_base *= 2;
      step--;
    }
  }

  // Use radix-4 all the way to the bottom, using different vectorization methods depending on step
  // size
  for (; step != 1; step -= 2) {
    radix4_high();
    k_base *= 4;
    if (n_pow >= FFT_RECURSIVE) {
      break;
    }
  }

  if (n_pow >= FFT_RECURSIVE) {
    fft_dif(real, imag, n_pow - 2, k_base);
    fft_dif(real, imag, n_pow - 2, k_base + 1);
    fft_dif(real, imag, n_pow - 2, k_base + 2);
    fft_dif(real, imag, n_pow - 2, k_base + 3);
  } else {
    radix4_low();
  }
}

void ifft_dif(double* real, double* imag, int n_pow, size_t k_base) {
  // This algorithm is a straightforward reverse of FFT-DIF, except that the result is multiplicated
  // by n:
  //     def IFFT-DIF(A):
  //         for step in range(0, log_2 n):
  //             for k in range(n / 2^{step+1}):
  //                 for j in range(2k * 2^step, (2k+1) * 2^step):
  //                     e, o, w = A[j], A[j+2^step], W_bitreversed[k]
  //                     A[j] = e + o
  //                     A[j+2^step] = (e - o) * conj(w)

  // Non-vectorized code:
  //     for (int step = 0; step < n_pow; step++) {
  //       for (size_t k = 0; k < (1uz << (n_pow - 1 - step)); k++) {
  //         double w_real = twiddles_bitreversed[k];
  //         double w_imag = twiddles_bitreversed[k + (1uz << (twiddles_n_pow - 1))];
  //         for (size_t j = (2 * k) << step; j < ((2 * k + 1) << step); j++) {
  //           double e_real = real[j];
  //           double e_imag = imag[j];
  //           double o_real = real[j + (1uz << step)];
  //           double o_imag = imag[j + (1uz << step)];
  //           double eo_real = e_real - o_real;
  //           double eo_imag = e_imag - o_imag;
  //           real[j] = e_real + o_real;
  //           imag[j] = e_imag + o_imag;
  //           real[j + (1uz << step)] = w_real * eo_real + w_imag * eo_imag;
  //           imag[j + (1uz << step)] = w_real * eo_imag - w_imag * eo_real;
  //         }
  //       }
  //     }

  int step = 0;

  // It's critical for performance to perform multiple steps at once
  auto radix4_high = [&]() {
    for (size_t k = k_base; k < k_base + (1uz << (n_pow - 2 - step)); k++) {
      __m256d w_real = _mm256_set1_pd(twiddles_bitreversed[k]);
      __m256d w_imag = _mm256_set1_pd(twiddles_bitreversed[k + (1uz << (twiddles_n_pow - 1))]);
      __m256d w0_real = _mm256_set1_pd(twiddles_bitreversed[2 * k]);
      __m256d w0_imag = _mm256_set1_pd(twiddles_bitreversed[2 * k + (1uz << (twiddles_n_pow - 1))]);
      __m256d w1_real = _mm256_set1_pd(twiddles_bitreversed[2 * k + 1]);
      __m256d w1_imag =
          _mm256_set1_pd(twiddles_bitreversed[2 * k + 1 + (1uz << (twiddles_n_pow - 1))]);

      for (size_t j = (4 * k) << step; j < ((4 * k + 1) << step); j += 4) {
        __m256d e0_real = _mm256_load_pd(&real[j]);
        __m256d e0_imag = _mm256_load_pd(&imag[j]);
        __m256d o0_real = _mm256_load_pd(&real[j + (1uz << step)]);
        __m256d o0_imag = _mm256_load_pd(&imag[j + (1uz << step)]);
        __m256d e1_real = _mm256_load_pd(&real[j + (2uz << step)]);
        __m256d e1_imag = _mm256_load_pd(&imag[j + (2uz << step)]);
        __m256d o1_real = _mm256_load_pd(&real[j + (3uz << step)]);
        __m256d o1_imag = _mm256_load_pd(&imag[j + (3uz << step)]);

        __m256d eo0_real = _mm256_sub_pd(e0_real, o0_real);
        __m256d eo0_imag = _mm256_sub_pd(e0_imag, o0_imag);
        __m256d eo1_real = _mm256_sub_pd(e1_real, o1_real);
        __m256d eo1_imag = _mm256_sub_pd(e1_imag, o1_imag);

        __m256d e2_real = _mm256_add_pd(e0_real, o0_real);
        __m256d e2_imag = _mm256_add_pd(e0_imag, o0_imag);
        __m256d e3_real = _mm256_fmadd_pd(w0_real, eo0_real, _mm256_mul_pd(w0_imag, eo0_imag));
        __m256d e3_imag = _mm256_fmsub_pd(w0_real, eo0_imag, _mm256_mul_pd(w0_imag, eo0_real));
        __m256d o2_real = _mm256_add_pd(e1_real, o1_real);
        __m256d o2_imag = _mm256_add_pd(e1_imag, o1_imag);
        __m256d o3_real = _mm256_fmadd_pd(w1_real, eo1_real, _mm256_mul_pd(w1_imag, eo1_imag));
        __m256d o3_imag = _mm256_fmsub_pd(w1_real, eo1_imag, _mm256_mul_pd(w1_imag, eo1_real));

        __m256d eo2_real = _mm256_sub_pd(e2_real, o2_real);
        __m256d eo2_imag = _mm256_sub_pd(e2_imag, o2_imag);
        __m256d eo3_real = _mm256_sub_pd(e3_real, o3_real);
        __m256d eo3_imag = _mm256_sub_pd(e3_imag, o3_imag);

        _mm256_store_pd(&real[j], _mm256_add_pd(e2_real, o2_real));
        _mm256_store_pd(&imag[j], _mm256_add_pd(e2_imag, o2_imag));
        _mm256_store_pd(&real[j + (1uz << step)], _mm256_add_pd(e3_real, o3_real));
        _mm256_store_pd(&imag[j + (1uz << step)], _mm256_add_pd(e3_imag, o3_imag));
        _mm256_store_pd(&real[j + (2uz << step)],
                        _mm256_fmadd_pd(w_real, eo2_real, _mm256_mul_pd(w_imag, eo2_imag)));
        _mm256_store_pd(&imag[j + (2uz << step)],
                        _mm256_fmsub_pd(w_real, eo2_imag, _mm256_mul_pd(w_imag, eo2_real)));
        _mm256_store_pd(&real[j + (3uz << step)],
                        _mm256_fmadd_pd(w_real, eo3_real, _mm256_mul_pd(w_imag, eo3_imag)));
        _mm256_store_pd(&imag[j + (3uz << step)],
                        _mm256_fmsub_pd(w_real, eo3_imag, _mm256_mul_pd(w_imag, eo3_real)));
      }
    }
  };

  auto radix4_low = [&]() {
    ensure(step == 0);

    for (size_t k = k_base; k < k_base + (1uz << (n_pow - 2)); k += 4) {
      __m256d w_real = _mm256_load_pd(&twiddles_bitreversed[k]);
      __m256d w_imag = _mm256_load_pd(&twiddles_bitreversed[k + (1uz << (twiddles_n_pow - 1))]);

      __m256d w01_real_low = _mm256_load_pd(&twiddles_bitreversed[2 * k]);
      __m256d w01_real_high = _mm256_load_pd(&twiddles_bitreversed[2 * k + 4]);
      __m256d w0_real =
          _mm256_permute4x64_pd(_mm256_unpacklo_pd(w01_real_low, w01_real_high), 0b11011000);
      __m256d w1_real =
          _mm256_permute4x64_pd(_mm256_unpackhi_pd(w01_real_low, w01_real_high), 0b11011000);

      __m256d w01_imag_low =
          _mm256_load_pd(&twiddles_bitreversed[2 * k + (1uz << (twiddles_n_pow - 1))]);
      __m256d w01_imag_high =
          _mm256_load_pd(&twiddles_bitreversed[2 * k + 4 + (1uz << (twiddles_n_pow - 1))]);
      __m256d w0_imag =
          _mm256_permute4x64_pd(_mm256_unpacklo_pd(w01_imag_low, w01_imag_high), 0b11011000);
      __m256d w1_imag =
          _mm256_permute4x64_pd(_mm256_unpackhi_pd(w01_imag_low, w01_imag_high), 0b11011000);

      auto [e0_real, o0_real, e1_real, o1_real] =
          transpose(_mm256_load_pd(&real[4 * k]), _mm256_load_pd(&real[4 * (k + 1)]),
                    _mm256_load_pd(&real[4 * (k + 2)]), _mm256_load_pd(&real[4 * (k + 3)]));
      auto [e0_imag, o0_imag, e1_imag, o1_imag] = transpose(
          _mm256_load_pd(&imag[4 * k]), _mm256_load_pd(&imag[4 * (k + 1)]),
          _mm256_load_pd(&imag[4 * (k + 2)]),
          _mm256_load_pd(&imag[4 * (k + 3)]));

      __m256d eo0_real = _mm256_sub_pd(e0_real, o0_real);
      __m256d eo0_imag = _mm256_sub_pd(e0_imag, o0_imag);
      __m256d eo1_real = _mm256_sub_pd(e1_real, o1_real);
      __m256d eo1_imag = _mm256_sub_pd(e1_imag, o1_imag);

      __m256d e2_real = _mm256_add_pd(e0_real, o0_real);
      __m256d e2_imag = _mm256_add_pd(e0_imag, o0_imag);
      __m256d e3_real = _mm256_fmadd_pd(w0_real, eo0_real, _mm256_mul_pd(w0_imag, eo0_imag));
      __m256d e3_imag = _mm256_fmsub_pd(w0_real, eo0_imag, _mm256_mul_pd(w0_imag, eo0_real));
      __m256d o2_real = _mm256_add_pd(e1_real, o1_real);
      __m256d o2_imag = _mm256_add_pd(e1_imag, o1_imag);
      __m256d o3_real = _mm256_fmadd_pd(w1_real, eo1_real, _mm256_mul_pd(w1_imag, eo1_imag));
      __m256d o3_imag = _mm256_fmsub_pd(w1_real, eo1_imag, _mm256_mul_pd(w1_imag, eo1_real));

      __m256d eo2_real = _mm256_sub_pd(e2_real, o2_real);
      __m256d eo2_imag = _mm256_sub_pd(e2_imag, o2_imag);
      __m256d eo3_real = _mm256_sub_pd(e3_real, o3_real);
      __m256d eo3_imag = _mm256_sub_pd(e3_imag, o3_imag);

      auto [b0_real, b1_real, b2_real, b3_real] =
          transpose(_mm256_add_pd(e2_real, o2_real), _mm256_add_pd(e3_real, o3_real),
                    _mm256_fmadd_pd(w_real, eo2_real, _mm256_mul_pd(w_imag, eo2_imag)),
                    _mm256_fmadd_pd(w_real, eo3_real, _mm256_mul_pd(w_imag, eo3_imag)));
      _mm256_store_pd(&real[4 * k], b0_real);
      _mm256_store_pd(&real[4 * (k + 1)], b1_real);
      _mm256_store_pd(&real[4 * (k + 2)], b2_real);
      _mm256_store_pd(&real[4 * (k + 3)], b3_real);

      auto [b0_imag, b1_imag, b2_imag, b3_imag] =
          transpose(_mm256_add_pd(e2_imag, o2_imag), _mm256_add_pd(e3_imag, o3_imag),
                    _mm256_fmsub_pd(w_real, eo2_imag, _mm256_mul_pd(w_imag, eo2_real)),
                    _mm256_fmsub_pd(w_real, eo3_imag, _mm256_mul_pd(w_imag, eo3_real)));
      _mm256_store_pd(&imag[4 * k], b0_imag);
      _mm256_store_pd(&imag[4 * (k + 1)], b1_imag);
      _mm256_store_pd(&imag[4 * (k + 2)], b2_imag);
      _mm256_store_pd(&imag[4 * (k + 3)], b3_imag);
    }
  };

  auto radix2_high = [&]() {
    for (size_t k = k_base; k < k_base + (1uz << (n_pow - 1 - step)); k++) {
      __m256d w_real = _mm256_set1_pd(twiddles_bitreversed[k]);
      __m256d w_imag = _mm256_set1_pd(twiddles_bitreversed[k + (1uz << (twiddles_n_pow - 1))]);
      for (size_t j = (2 * k) << step; j < ((2 * k + 1) << step); j += 4) {
        __m256d e_real = _mm256_load_pd(&real[j]);
        __m256d e_imag = _mm256_load_pd(&imag[j]);
        __m256d o_real = _mm256_load_pd(&real[j + (1uz << step)]);
        __m256d o_imag = _mm256_load_pd(&imag[j + (1uz << step)]);
        __m256d eo_real = _mm256_sub_pd(e_real, o_real);
        __m256d eo_imag = _mm256_sub_pd(e_imag, o_imag);
        _mm256_store_pd(&real[j], _mm256_add_pd(e_real, o_real));
        _mm256_store_pd(&imag[j], _mm256_add_pd(e_imag, o_imag));
        _mm256_store_pd(&real[j + (1uz << step)],
                        _mm256_fmadd_pd(w_real, eo_real, _mm256_mul_pd(w_imag, eo_imag)));
        _mm256_store_pd(&imag[j + (1uz << step)],
                        _mm256_fmsub_pd(w_real, eo_imag, _mm256_mul_pd(w_imag, eo_real)));
      }
    }
  };

  // If n_pow >= FFT_RECURSIVE, we want to do recurse and then do one radix-4 step. Otherwise, apply
  // various steps iteratively. We want to only have one invocation of radix4_high so that it is
  // inlined, so the code is a bit more messy than it could be.

  if (n_pow >= FFT_RECURSIVE) {
    ifft_dif(real, imag, n_pow - 2, k_base);
    ifft_dif(real, imag, n_pow - 2, k_base + (1uz << (n_pow - 2)));
    ifft_dif(real, imag, n_pow - 2, k_base + (2uz << (n_pow - 2)));
    ifft_dif(real, imag, n_pow - 2, k_base + (3uz << (n_pow - 2)));
    step = n_pow - 2;
    k_base >>= n_pow - 2;
  } else {
    k_base /= 4;
    radix4_low();
    step += 2;
  }

  for (; step + 2 <= n_pow; step += 2) {
    k_base /= 4;
    radix4_high();
  }

  if (step < n_pow) {
    k_base /= 2;
    radix2_high();
  }
}

__m256d u64_to_double(__m256i integers) {
  __m256d magic = _mm256_set1_pd(0x1p52);
  return _mm256_sub_pd(_mm256_xor_pd(_mm256_castsi256_pd(integers), magic), magic);
}

double* mul_fft_transform_input(ConstRef input, int n_pow, int word_bitness) {
  size_t n = 1uz << n_pow;

  // Add one double between real and imag halves to help cache aliasing
  double* input_fft = new (std::align_val_t(32)) double[n * 2 + 4];
  size_t imag_offset = n + 4;

  // Split into words
  size_t k = 0;
  if (word_bitness == 16) {
    for (; k < input.data.size(); k++) {
      __m256d fp = u64_to_double(_mm256_blend_epi16(_mm256_setzero_si256(), _mm256_srlv_epi64(_mm256_set1_epi64x(input.data[k]), _mm256_set_epi64x(48, 16, 32, 0)), 0x11));
      _mm_store_pd(&input_fft[k * 2], _mm256_castpd256_pd128(fp));
      _mm_store_pd(&input_fft[k * 2 + imag_offset], _mm256_extractf128_pd(fp, 1));
    }
  } else if (word_bitness == 12) {
    const uint16_t *data = reinterpret_cast<const uint16_t*>(input.data.data());
    for (; (k + 1) * 3 < input.data.size() * 4; k++) {
      // 48-bit word; it is safe to overread by 1 byte, and we mask it later with shifted 0xfff, so
      // no need to get rid of the top 16 bits
      uint64_t word;
      std::memcpy(&word, &data[k * 3], 8);
      // Split into 12-bit parts
      __m256d fp = u64_to_double(_mm256_and_si256(_mm256_srlv_epi64(_mm256_set1_epi64x(word), _mm256_set_epi64x(36, 12, 24, 0)), _mm256_set1_epi64x(0xfff)));
      _mm_store_pd(&input_fft[k * 2], _mm256_castpd256_pd128(fp));
      _mm_store_pd(&input_fft[k * 2 + imag_offset], _mm256_extractf128_pd(fp, 1));
    }

    // Last part:
    // 48-bit word
    uint64_t word = k * 3 + 2 < input.data.size() * 4 ? data[k * 3 + 2] : 0;
    word = (word << 16) | (k * 3 + 1 < input.data.size() * 4 ? data[k * 3 + 1] : 0);
    word = (word << 16) | data[k * 3];
    // Split into 12-bit parts
    __m256d fp = u64_to_double(_mm256_and_si256(_mm256_srlv_epi64(_mm256_set1_epi64x(word), _mm256_set_epi64x(36, 12, 24, 0)), _mm256_set1_epi64x(0xfff)));
    _mm_store_pd(&input_fft[k * 2], _mm256_castpd256_pd128(fp));
    _mm_store_pd(&input_fft[k * 2 + imag_offset], _mm256_extractf128_pd(fp, 1));
    k++;
  } else {
    ensure(false);
  }

  size_t count_left = n - k * 2;
  memzero64(reinterpret_cast<uint64_t*>(input_fft + k * 2), count_left);
  memzero64(reinterpret_cast<uint64_t*>(input_fft + k * 2 + imag_offset), count_left);

  fft_dif(input_fft, input_fft + imag_offset, n_pow, 0);

  return input_fft;
}

double* mul_fft_middle_end(double* lhs_fft_dif, double* rhs_fft_dif, int n_pow) {
  size_t n = 1uz << n_pow;

  // Add one double between real and imag halves to help cache aliasing
  double* prod_fft_dif = new (std::align_val_t(32)) double[n * 2 + 4];
  size_t imag_offset = n + 4;

  auto handle_iteration_single = [&](size_t k, size_t k_complement) {
    double lhs_k_real = lhs_fft_dif[k];
    double lhs_k_imag = lhs_fft_dif[k + imag_offset];
    double rhs_k_real = rhs_fft_dif[k];
    double rhs_k_imag = rhs_fft_dif[k + imag_offset];
    double twiddles_bitreversed_k_real = twiddles_bitreversed[k / 2];
    double twiddles_bitreversed_k_imag =
        twiddles_bitreversed[k / 2 + (1uz << (twiddles_n_pow - 1))];
    double sign = k % 2 == 1 ? -1 : 1;

    double a_real = lhs_k_real - lhs_fft_dif[k_complement];
    double a_imag = lhs_k_imag + lhs_fft_dif[k_complement + imag_offset];
    double b_real = rhs_k_real - rhs_fft_dif[k_complement];
    double b_imag = rhs_k_imag + rhs_fft_dif[k_complement + imag_offset];
    double c_real = a_real * b_real - a_imag * b_imag;
    double c_imag = a_real * b_imag + a_imag * b_real;

    prod_fft_dif[k] = ((lhs_k_real * rhs_k_real - lhs_k_imag * rhs_k_imag) -
                       ((1 + sign * twiddles_bitreversed_k_real) * c_real -
                        sign * twiddles_bitreversed_k_imag * c_imag) /
                           4);
    prod_fft_dif[k + imag_offset] = ((lhs_k_real * rhs_k_imag + lhs_k_imag * rhs_k_real) -
                           ((1 + sign * twiddles_bitreversed_k_real) * c_imag +
                            sign * twiddles_bitreversed_k_imag * c_real) /
                               4);
  };

  double* twiddles_bitreversed_ptr = twiddles_bitreversed;
  size_t twiddles_imag_offset = 1uz << (twiddles_n_pow - 1);

  auto handle_iteration_vectorized = [&](size_t k, size_t k_complement) {
    __m256d lhs_k_real = _mm256_load_pd(&lhs_fft_dif[k]);
    __m256d lhs_k_imag = _mm256_load_pd(&lhs_fft_dif[k + imag_offset]);
    __m256d rhs_k_real = _mm256_load_pd(&rhs_fft_dif[k]);
    __m256d rhs_k_imag = _mm256_load_pd(&rhs_fft_dif[k + imag_offset]);

    __m256d twiddles_bitreversed_k_real = _mm256_permute4x64_pd(
        _mm256_castpd128_pd256(_mm_load_pd(&twiddles_bitreversed_ptr[k / 2])), 0x50);
    __m256d twiddles_bitreversed_k_imag =
        _mm256_permute4x64_pd(_mm256_castpd128_pd256(_mm_load_pd(
                                  &twiddles_bitreversed_ptr[k / 2 + twiddles_imag_offset])),
                              0x50);

    __m256d a_real = _mm256_sub_pd(
        lhs_k_real, _mm256_permute4x64_pd(_mm256_load_pd(&lhs_fft_dif[k_complement]), 0b00011011));
    __m256d a_imag = _mm256_add_pd(
        lhs_k_imag,
        _mm256_permute4x64_pd(_mm256_load_pd(&lhs_fft_dif[k_complement + imag_offset]), 0b00011011));
    __m256d b_real = _mm256_sub_pd(
        rhs_k_real, _mm256_permute4x64_pd(_mm256_load_pd(&rhs_fft_dif[k_complement]), 0b00011011));
    __m256d b_imag = _mm256_add_pd(
        rhs_k_imag,
        _mm256_permute4x64_pd(_mm256_load_pd(&rhs_fft_dif[k_complement + imag_offset]), 0b00011011));
    __m256d c_real = _mm256_fmsub_pd(a_real, b_real, _mm256_mul_pd(a_imag, b_imag));
    __m256d c_imag = _mm256_fmadd_pd(a_real, b_imag, _mm256_mul_pd(a_imag, b_real));

    _mm256_store_pd(
        &prod_fft_dif[k],
        _mm256_fmsub_pd(
            lhs_k_real, rhs_k_real,
            _mm256_fmsubadd_pd(
                lhs_k_imag, rhs_k_imag,
                _mm256_mul_pd(_mm256_set1_pd(0.25),
                              _mm256_fmsub_pd(twiddles_bitreversed_k_real, c_real,
                                              _mm256_fmaddsub_pd(twiddles_bitreversed_k_imag,
                                                                 c_imag, c_real))))));
    _mm256_store_pd(
        &prod_fft_dif[k + imag_offset],
        _mm256_fmadd_pd(
            lhs_k_real, rhs_k_imag,
            _mm256_fmaddsub_pd(
                lhs_k_imag, rhs_k_real,
                _mm256_mul_pd(_mm256_set1_pd(0.25),
                              _mm256_fmadd_pd(twiddles_bitreversed_k_real, c_imag,
                                              _mm256_fmsubadd_pd(twiddles_bitreversed_k_imag,
                                                                 c_real, c_imag))))));
  };

  handle_iteration_single(0, 0);
  handle_iteration_single(1, 1);
  handle_iteration_single(2, 3);
  handle_iteration_single(3, 2);
  for (int j = 2; j < n_pow; j++) {
    for (size_t k = (1uz << j); k < (2uz << j); k += 4) {
      size_t k_complement = (3uz << j) - 4 - k;
      handle_iteration_vectorized(k, k_complement);
    }
  }

  return prod_fft_dif;
}

void mul_fft_transform_output(Ref result, double* prod_fft_dif, int n_pow, int word_bitness) {
  size_t n = 1uz << n_pow;
  size_t imag_offset = n + 4;
  __m256d magic = _mm256_set1_pd(0x1p52 * n);

  ifft_dif(prod_fft_dif, prod_fft_dif + imag_offset, n_pow, 0);

  uint64_t carry = 0;

  auto get_word = [&](size_t k) {
    __m128d fp02 = _mm_load_pd(&prod_fft_dif[2 * k]);
    __m128d fp13 = _mm_load_pd(&prod_fft_dif[2 * k + imag_offset]);
    __m256d fp = _mm256_set_m128d(_mm_unpackhi_pd(fp02, fp13), _mm_unpacklo_pd(fp02, fp13));
    return _mm256_castpd_si256(
        _mm256_xor_pd(_mm256_add_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.), fp), magic), magic));
  };

  if (word_bitness == 16) {
    for (size_t k = 0; k < result.data.size(); k++) {
      __m256i word = get_word(k);
      __uint128_t tmp = static_cast<uint64_t>(word[3]);
      tmp = (tmp << 16) + static_cast<uint64_t>(word[2]);
      tmp = (tmp << 16) + static_cast<uint64_t>(word[1]);
      tmp = (tmp << 16) + static_cast<uint64_t>(word[0]);
      tmp += carry;
      result.data[k] = static_cast<uint64_t>(tmp);
      carry = static_cast<uint64_t>(tmp >> 64);
    }
  } else if (word_bitness == 12) {
    uint16_t *data = reinterpret_cast<uint16_t*>(result.data.data());
    size_t k = 0;
    for (; (k + 1) * 3 < result.data.size() * 4; k++) {
      __m256i word = get_word(k);
      __uint128_t tmp = static_cast<uint64_t>(word[3]);
      tmp = (tmp << 12) + static_cast<uint64_t>(word[2]);
      tmp = (tmp << 12) + static_cast<uint64_t>(word[1]);
      tmp = (tmp << 12) + static_cast<uint64_t>(word[0]);
      tmp += carry;
      memcpy(&data[k * 3], &tmp, 6);
      carry = static_cast<uint64_t>(tmp >> 48);
    }

    // Last part:
    __m256i word = get_word(k);
    __uint128_t tmp = static_cast<uint64_t>(word[3]);
    tmp = (tmp << 12) + static_cast<uint64_t>(word[2]);
    tmp = (tmp << 12) + static_cast<uint64_t>(word[1]);
    tmp = (tmp << 12) + static_cast<uint64_t>(word[0]);
    tmp += carry;
    data[k * 3] = static_cast<uint16_t>(tmp);
    if (k * 3 + 2 < result.data.size() * 4) {
      data[k * 3 + 1] = static_cast<uint16_t>(tmp >> 16);
      data[k * 3 + 2] = static_cast<uint16_t>(tmp >> 32);
      carry = static_cast<uint64_t>(tmp >> 48);
    } else if (k * 3 + 1 < result.data.size() * 4) {
      data[k * 3 + 1] = static_cast<uint16_t>(tmp >> 16);
      carry = static_cast<uint64_t>(tmp >> 32);
    } else {
      carry = static_cast<uint64_t>(tmp >> 16);
    }
  } else {
    ensure(false);
  }

  ensure(carry == 0);
}

std::ostream& operator<<(std::ostream& out, ConstRef rhs);

void mul_fft(Ref result, ConstRef lhs, ConstRef rhs, int n_pow, int word_bitness) {
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
  // Anyway, suppose the functions we actually have are FFT-DIF and IFFT-DIF (decimation in
  // frequency), where
  //     FFT-DIF[P]_k = FFT[P]_{bitreverse(k)}
  //     IFFT-DIF[A] = n IFFT[j | A_{bitreverse(j)}]
  // These functions are both easier to compute than FFT. They also form "inverses":
  //     IFFT-DIF[FFT-DIF[P]] = n P
  //     FFT-DIF[IFFT-DIF[A]] = n A
  //
  // To utilize the inverse formula, we can rewrite the steps to do the following:
  // 1. Transform P1(x) and P2(x) to Q1(x) and Q2(x)
  // 2. Compute FFT-DIF[Q1] and FFT-DIF[Q2]
  // 3. Transform FFT-DIF[Q1] and FFT-DIF[Q2] to FFT-DIF[P1] and FFT-DIF[P2]
  // 4. Obtain FFT-DIF[Pr] by pointwise multiplication
  // 5. Transform FFT-DIF[Pr] to FFT-DIF[Qr]
  // 6. Compute IFFT-DIF[FFT-DIF[Qr]] = n Qr(x)
  // 7. Transform n Qr(x) to Pr(x)
  //
  // Thus steps 3-5 are rewritten like this:
  //     FFT-DIF[Qr]_k = FFT-DIF[Q1]_k FFT-DIF[Q2]_k - (
  //                         (1 + w_{n/2}^{bitreverse(k)})
  //                         * (FFT-DIF[Q1]_k - conj(FFT-DIF[Q1]_{bitreverse(-bitreverse(k))}))
  //                         * (FFT-DIF[Q2]_k - conj(FFT-DIF[Q2]_{bitreverse(-bitreverse(k))}))
  //                         / 4
  //                     )
  // It's useful to consider how cache-efficient it is to access indices bitreverse(-bitreverse(k))
  // for k = 0, 1, 2... As -x = ~x + 1, we have
  //     bitreverse(-bitreverse(k)) = bitreverse(bitreverse(~k) + 1),
  // i.e. incrementing ~k at its highest digit in reverse bitorder (as opposed to classic addition).
  // For k = 00...01 || A, we have ~k = 11...10 || ~A, so the formula maps k to 00...01 || ~A. Thus,
  // for each j, the formula reverses the [2^j; 2^{j+1}-1] segment. The sequence is hence:
  //     0; 1; 3, 2; 7, 6, 5, 4; 15, 14, 13, 12, 11, 10, 9, 8; ...
  // This also gives a way to interate through k and obtain the index without any additional
  // computations. As there are O(log n) such segments, there should only be O(log n) cache misses
  // before the predictor aligns to the next segment.

  ensure_twiddle_factors(n_pow);
  double* lhs_fft_dif = mul_fft_transform_input(lhs, n_pow, word_bitness);
  double* rhs_fft_dif = mul_fft_transform_input(rhs, n_pow, word_bitness);
  double* prod_fft_dif = mul_fft_middle_end(lhs_fft_dif, rhs_fft_dif, n_pow);
  ::operator delete[](lhs_fft_dif, std::align_val_t(32));
  ::operator delete[](rhs_fft_dif, std::align_val_t(32));
  mul_fft_transform_output(result.slice(0, lhs.data.size() + rhs.data.size()), prod_fft_dif, n_pow, word_bitness);
  ::operator delete[](prod_fft_dif, std::align_val_t(32));
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

BigInt::BigInt(const char* s, with_base base) : BigInt(std::string_view(s), base) {}

BigInt::BigInt(ConstRef rhs) : data(rhs.data) {}
BigInt::BigInt(const BigInt& rhs) : data(rhs.data) {}
BigInt::BigInt(BigInt&& rhs) : data(std::move(rhs.data)) {}

BigInt& BigInt::operator=(ConstRef rhs) {
  data = rhs.data;
  return *this;
}

BigInt& BigInt::operator=(BigInt&& rhs) {
  data = std::move(rhs.data);
  return *this;
}

bool operator==(ConstRef lhs, ConstRef rhs) { return lhs.data == rhs.data; }
bool operator!=(ConstRef lhs, ConstRef rhs) { return !(lhs == rhs); }

bool operator<(ConstRef lhs, ConstRef rhs) {
  if (lhs.data.size() != rhs.data.size()) {
    return lhs.data.size() < rhs.data.size();
  }
  return std::lexicographical_compare(lhs.data.rbegin(), lhs.data.rend(), rhs.data.rbegin(),
                                      rhs.data.rend());
}
bool operator>(ConstRef lhs, ConstRef rhs) { return rhs < lhs; }
bool operator<=(ConstRef lhs, ConstRef rhs) { return !(rhs < lhs); }
bool operator>=(ConstRef lhs, ConstRef rhs) { return !(lhs < rhs); }

BigInt& BigInt::operator+=(ConstRef rhs) {
  data.increase_size_zerofill(std::max(data.size(), rhs.data.size()) + 1);
  add_to(*this, rhs);
  if (data.back() == 0) {
    data.pop_back();
  }
  return *this;
}

BigInt& BigInt::operator-=(ConstRef rhs) {
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
        [unrolled_loop_count] "+r"(unrolled_loop_count), [left_loop_count] "+r"(left_loop_count)
      : [data_ptr] "r"(data.data()), [rhs_data_ptr] "r"(rhs.data.data())
      : "flags", "memory");

  _normalize();
  return *this;
}

BigInt& BigInt::operator+=(const BigInt& rhs) { return *this += static_cast<ConstRef>(rhs); }
BigInt& BigInt::operator-=(const BigInt& rhs) { return *this -= static_cast<ConstRef>(rhs); }

BigInt& BigInt::operator++() { return *this += 1; }
BigInt BigInt::operator++(int) {
  BigInt tmp = *this;
  ++*this;
  return tmp;
}

BigInt& BigInt::operator--() { return *this -= 1; }
BigInt BigInt::operator--(int) {
  BigInt tmp = *this;
  --*this;
  return tmp;
}

BigInt operator+(BigInt lhs, ConstRef rhs) { return lhs += rhs; }
BigInt operator-(BigInt lhs, ConstRef rhs) { return lhs -= rhs; }

BigInt& operator*=(BigInt& lhs, uint64_t rhs) {
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
  uint32_t* lhs_data = reinterpret_cast<uint32_t*>(data.data());
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

BigInt& operator/=(BigInt& lhs, uint32_t rhs) {
  lhs.divmod_inplace(rhs);
  return lhs;
}

uint32_t operator%(ConstRef lhs, uint32_t rhs) {
  const uint32_t* data = reinterpret_cast<const uint32_t*>(lhs.data.data());
  uint32_t remainder = 0;
  for (size_t i = lhs.data.size() * 2; i > 0; i--) {
    uint64_t cur = data[i - 1] | (uint64_t{remainder} << 32);
    remainder = cur % rhs;
  }
  return remainder;
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
  result.data[1] = static_cast<uint64_t>(product >> 64);
}

void mul_nx1(Ref result, ConstRef lhs, uint64_t rhs) {
  uint64_t carry = 0;
  for (size_t i = 0; i < lhs.data.size(); i++) {
    __uint128_t total = __uint128_t{lhs.data[i]} * rhs + carry;
    result.data[i] = static_cast<uint64_t>(total);
    carry = static_cast<uint64_t>(total >> 64);
  }
  result.data[lhs.data.size()] = carry;
}

__attribute__((noinline)) void mul_quadratic(Ref result, ConstRef lhs, ConstRef rhs) {
  size_t size = lhs.data.size() + rhs.data.size() - 1;

  uint64_t carry_low = 0;
  uint64_t carry_high = 0;
  for (size_t i = 0; i < size; i++) {
    size_t left = static_cast<size_t>(std::max(static_cast<ssize_t>(i + 1 - rhs.data.size()), 0z));
    size_t right = std::min(i + 1, lhs.data.size());

    uint64_t sum_low = carry_low;
    uint64_t sum_mid = carry_high;
    uint64_t sum_high = 0;

#define LOOP                                                                                       \
  do {                                                                                             \
    uint64_t rax = lhs.data[left];                                                                 \
    asm("mulq %[b];"                                                                               \
        "add %%rax, %[sum_low];"                                                                   \
        "adc %%rdx, %[sum_mid];"                                                                   \
        "adc $0, %[sum_high];"                                                                     \
        : "+a"(rax), [sum_low] "+r"(sum_low), [sum_mid] "+r"(sum_mid), [sum_high] "+r"(sum_high)   \
        : [b] "m"(rhs.data[i - left])                                                              \
        : "flags", "rdx");                                                                         \
    left++;                                                                                        \
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
  BigInt r2 =
      (std::move(a0_plus_a1_plus_a2) + a1 + a2 * 3) * (std::move(b0_plus_b1_plus_b2) + b1 + b2 * 3);
  BigInt rinf_2 = rinf * 2;

  auto half = [](BigInt a) {
    ensure(!a.halve());
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
    int n_pow = get_fft_n_pow_16bit(lhs, rhs);
    if (n_pow >= FFT_CUTOFF) {
      // Large enough to be efficient
      if (n_pow <= FFT_MAX_16BIT) {
        // Small enough to be precise if input is split into 16-bit words
        mul_fft(result, lhs, rhs, n_pow, 16);
      } else {
        n_pow = get_fft_n_pow_12bit(lhs, rhs);
        ensure(n_pow <= FFT_MAX_12BIT);  // XXX: is this impossible?
        mul_fft(result, lhs, rhs, n_pow, 12);
      }
    } else if (lhs.data.size() * 2 < rhs.data.size()) {
      mul_disproportional(result, lhs, rhs);
    } else if (rhs.data.size() * 2 < lhs.data.size()) {
      mul_disproportional(result, rhs, lhs);
    } else {
      mul_karatsuba(result, lhs, rhs);
    }
  } else {
    mul_quadratic(result, lhs, rhs);
  }

  // ensure((lhs % 179) * (rhs % 179) % 179 == result.slice(0, lhs.data.size() + rhs.data.size()) % 179);
}

BigInt operator*(ConstRef lhs, ConstRef rhs) {
  if (lhs.data.empty() || rhs.data.empty()) {
    return {};
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
__uint128_t str_to_int_128(Iterator begin, Iterator end, uint64_t base, int max_block_len,
                           uint64_t base_product, Map map) {
  uint64_t low = str_to_int_64(begin, begin + max_block_len, base, map);
  uint64_t high = str_to_int_64(begin + max_block_len, end, base, map);
  return static_cast<__uint128_t>(high) * base_product + low;
}

template <typename Iterator, typename Map>
void str_to_int_inplace(Iterator begin, Iterator end, uint64_t base, Map map,
                        const BigInt* powers_of_base, int max_block_len, uint64_t base_product,
                        BigInt& result) {
  if (end - begin <= max_block_len) {
    result += str_to_int_64(begin, end, base, map);
    return;
  } else if (end - begin <= 2 * max_block_len) {
    result += str_to_int_128(begin, end, base, max_block_len, base_product, map);
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

  int low_len_pow = 63 - __builtin_clzll(static_cast<uint64_t>(end - begin - 1));
  ssize_t low_len = 1z << low_len_pow;
  Iterator mid = begin + low_len;
  BigInt high;
  str_to_int_inplace(mid, end, base, map, powers_of_base, max_block_len, base_product, high);
  result += high * powers_of_base[low_len_pow];
  str_to_int_inplace(begin, mid, base, map, powers_of_base, max_block_len, base_product, result);
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
  while ((1z << powers_of_base.size()) <= end - begin) {
    powers_of_base.push_back(powers_of_base.back() * powers_of_base.back());
  }

  BigInt result;
  str_to_int_inplace(begin, end, base, map, powers_of_base.data(), max_block_len, base_product,
                     result);
  return result;
}

template <typename List, typename> BigInt::BigInt(List&& list, with_base base) {
  *this = str_to_int(list.begin(), list.end(), base.base, [](uint64_t digit) { return digit; });
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

std::ostream& operator<<(std::ostream& out, ConstRef rhs) {
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

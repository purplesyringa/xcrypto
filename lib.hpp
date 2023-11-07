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
  Ref normalize_nonzero() {
    Span tmp = data;
    while (tmp.back() == 0) {
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
  ConstRef normalize_nonzero() {
    ConstSpan tmp = data;
    while (tmp.back() == 0) {
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

inline auto cosines = std::make_unique<double[]>(1);
inline int cosines_n_pow = -1;

// j = -i
__m256d mul_by_j(__m256d vec) {
  return _mm256_xor_pd(_mm256_permute_pd(vec, 5),
                       _mm256_set_pd(-0., 0., -0., 0.));
}

__m256d load_w(size_t n, size_t cos, size_t sin) {
  __m128d reals = _mm_load_pd(&cosines[n + cos]);
  __m128d imags = _mm_load_pd(&cosines[n + sin]);
  return _mm256_set_m128d(_mm_unpackhi_pd(reals, imags),
                          _mm_unpacklo_pd(reals, imags));
}

__m256d mul(__m256d a, __m256d b) {
  return _mm256_fmaddsub_pd(
      _mm256_movedup_pd(a), b,
      _mm256_mul_pd(_mm256_permute_pd(a, 15), _mm256_permute_pd(b, 5)));
}

using Complex = double[2];

void fft_cooley_tukey_no_transpose_4(Complex *data, int n_pow) {
  size_t old_n = size_t{1} << n_pow;

  while (n_pow > 2) {
    size_t n = size_t{1} << n_pow;
    size_t n2 = size_t{1} << (n_pow - 2);

    for (Complex *cur_data = data; cur_data != data + old_n; cur_data += n) {
      for (size_t i = 0; i < n2; i += 2) {
        __m256d a0 = _mm256_load_pd(cur_data[i]);
        __m256d a1 = _mm256_load_pd(cur_data[n2 + i]);
        __m256d a2 = _mm256_load_pd(cur_data[n2 * 2 + i]);
        __m256d a3 = _mm256_load_pd(cur_data[n2 * 3 + i]);

        __m256d c0 = _mm256_add_pd(a0, a2);
        __m256d c1 = _mm256_add_pd(a1, a3);
        __m256d c2 = _mm256_sub_pd(a0, a2);
        __m256d c3 = mul_by_j(_mm256_sub_pd(a1, a3));

        __m256d b0 = _mm256_add_pd(c0, c1);
        __m256d b1 = _mm256_add_pd(c2, c3);
        __m256d b2 = _mm256_sub_pd(c0, c1);
        __m256d b3 = _mm256_sub_pd(c2, c3);

        __m256d w1 = load_w(n, i, i + n / 4);
        __m256d w2 = load_w(n / 2, i, i + n / 8);
        __m256d w3 = mul(w1, w2);

        _mm256_store_pd(cur_data[i], b0);
        _mm256_store_pd(cur_data[n2 + i], mul(w1, b1));
        _mm256_store_pd(cur_data[n2 * 2 + i], mul(w2, b2));
        _mm256_store_pd(cur_data[n2 * 3 + i], mul(w3, b3));
      }
    }

    n_pow -= 2;
  }

  for (Complex *cur_data = data; cur_data != data + old_n; cur_data += 4) {
    __m256d a01 = _mm256_load_pd(cur_data[0]);
    __m256d a23 = _mm256_load_pd(cur_data[2]);

    __m256d c01 = _mm256_add_pd(a01, a23);
    __m256d c23 = _mm256_xor_pd(_mm256_permute_pd(_mm256_sub_pd(a01, a23), 6),
                                _mm256_set_pd(-0., 0., 0., 0.));

    __m256d c02 = _mm256_permute2f128_pd(c01, c23, 0x20);
    __m256d c13 = _mm256_permute2f128_pd(c01, c23, 0x31);

    __m256d b01 = _mm256_add_pd(c02, c13);
    __m256d b23 = _mm256_sub_pd(c02, c13);

    _mm256_store_pd(cur_data[0], b01);
    _mm256_store_pd(cur_data[2], b23);
  }
}

void fft_cooley_tukey_no_transpose_8(Complex *data, int n_pow, int count3) {
  if (count3 == 0) {
    fft_cooley_tukey_no_transpose_4(data, n_pow);
    return;
  }

  static const __m256d rsqrt2 = _mm256_set1_pd(1. / std::sqrt(2.));

  size_t n = size_t{1} << n_pow;
  size_t n2 = size_t{1} << (n_pow - 3);

  for (size_t i = 0; i < n2; i += 2) {
    __m256d a0 = _mm256_load_pd(data[i]);
    __m256d a1 = _mm256_load_pd(data[n2 + i]);
    __m256d a2 = _mm256_load_pd(data[n2 * 2 + i]);
    __m256d a3 = _mm256_load_pd(data[n2 * 3 + i]);
    __m256d a4 = _mm256_load_pd(data[n2 * 4 + i]);
    __m256d a5 = _mm256_load_pd(data[n2 * 5 + i]);
    __m256d a6 = _mm256_load_pd(data[n2 * 6 + i]);
    __m256d a7 = _mm256_load_pd(data[n2 * 7 + i]);

    __m256d e0 = _mm256_add_pd(a0, a4);
    __m256d e1 = _mm256_sub_pd(a0, a4);

    __m256d f0 = _mm256_add_pd(a2, a6);
    __m256d f1 = mul_by_j(_mm256_sub_pd(a2, a6));

    __m256d c0 = _mm256_add_pd(e0, f0);
    __m256d c1 = _mm256_add_pd(e1, f1);
    __m256d c2 = _mm256_sub_pd(e0, f0);
    __m256d c3 = _mm256_sub_pd(e1, f1);

    __m256d g0 = _mm256_add_pd(a1, a5);
    __m256d g1 = _mm256_sub_pd(a1, a5);

    __m256d h0 = _mm256_add_pd(a3, a7);
    __m256d h1 = mul_by_j(_mm256_sub_pd(a3, a7));

    __m256d k0 = _mm256_mul_pd(_mm256_add_pd(g1, h1), rsqrt2);
    __m256d k1 = _mm256_mul_pd(_mm256_sub_pd(g1, h1), rsqrt2);

    __m256d d0 = _mm256_add_pd(g0, h0);
    __m256d d1 = _mm256_add_pd(mul_by_j(k0), k0);
    __m256d d2 = mul_by_j(_mm256_sub_pd(g0, h0));
    __m256d d3 = _mm256_sub_pd(mul_by_j(k1), k1);

    __m256d b0 = _mm256_add_pd(c0, d0);
    __m256d b1 = _mm256_add_pd(c1, d1);
    __m256d b2 = _mm256_add_pd(c2, d2);
    __m256d b3 = _mm256_add_pd(c3, d3);
    __m256d b4 = _mm256_sub_pd(c0, d0);
    __m256d b5 = _mm256_sub_pd(c1, d1);
    __m256d b6 = _mm256_sub_pd(c2, d2);
    __m256d b7 = _mm256_sub_pd(c3, d3);

    __m256d w1 = load_w(n, i, i + n / 4);
    __m256d w2 = load_w(n / 2, i, i + n / 8);
    __m256d w3 = mul(w1, w2);
    __m256d w4 = load_w(n / 4, i, i + n / 16);
    __m256d w5 = mul(w1, w4);
    __m256d w6 = mul(w3, w3);
    __m256d w7 = mul(w4, w3);

    _mm256_store_pd(data[i], b0);
    _mm256_store_pd(data[n2 + i], mul(w1, b1));
    _mm256_store_pd(data[n2 * 2 + i], mul(w2, b2));
    _mm256_store_pd(data[n2 * 3 + i], mul(w3, b3));
    _mm256_store_pd(data[n2 * 4 + i], mul(w4, b4));
    _mm256_store_pd(data[n2 * 5 + i], mul(w5, b5));
    _mm256_store_pd(data[n2 * 6 + i], mul(w6, b6));
    _mm256_store_pd(data[n2 * 7 + i], mul(w7, b7));
  }

  for (size_t offset = 0; offset < n; offset += n2) {
    fft_cooley_tukey_no_transpose_8(data + offset, n_pow - 3, count3 - 1);
  }
}

inline constexpr int FFT_CUTOFF = 14;
inline constexpr int CT8_CUTOFF = 15;
inline constexpr int FFT_MIN =
    FFT_CUTOFF - 1; // -1 due to real-fft size halving optimization
inline constexpr int FFT_MAX = 19;

constexpr std::pair<int, int> get_counts(int n_pow) {
  int count3 = 0;
  while (n_pow > CT8_CUTOFF) {
    n_pow -= 3;
    count3++;
  }
  if (n_pow % 2 == 1) {
    n_pow -= 3;
    count3++;
  }
  int count2 = n_pow / 2;
  return {count3, count2};
}

__attribute__((always_inline)) inline __m256i shiftl(__m256i x, int shift) {
  if (shift > 0) {
    return _mm256_slli_epi32(x, shift);
  } else {
    return _mm256_srli_epi32(x, -shift);
  }
}

template <int N_POW, int... Iterator3, int... Iterator2>
__attribute__((always_inline)) inline __m256i
reverse_mixed_radix_const256_impl(__m256i number,
                                  std::integer_sequence<int, Iterator3...>,
                                  std::integer_sequence<int, Iterator2...>) {
  static constexpr int COUNT3 = get_counts(N_POW).first;
  __m256i result = _mm256_setzero_si256();
  ((result = _mm256_or_si256(
        result,
        shiftl(_mm256_and_si256(
                   number, _mm256_set1_epi32(uint32_t{7} << (Iterator3 * 3))),
               N_POW - (Iterator3 * 2 + 1) * 3))),
   ...);
  ((result = _mm256_or_si256(
        result,
        shiftl(_mm256_and_si256(
                   number, _mm256_set1_epi32(uint32_t{3}
                                             << (COUNT3 * 3 + Iterator2 * 2))),
               N_POW - COUNT3 * 3 * 2 - (Iterator2 * 2 + 1) * 2))),
   ...);
  return result;
}
template <int N_POW> __m256i reverse_mixed_radix_const256(__m256i number) {
  static constexpr int COUNT3 = get_counts(N_POW).first;
  static constexpr int COUNT2 = get_counts(N_POW).second;
  return reverse_mixed_radix_const256_impl<N_POW>(
      number, std::make_integer_sequence<int, COUNT3>(),
      std::make_integer_sequence<int, COUNT2>());
}

template <int... Pows>
__m256i reverse_mixed_radix_dyn(int n_pow, __m256i vec,
                                std::integer_sequence<int, Pows...>) {
  static constexpr __m256i (*dispatch[])(__m256i) = {
      &reverse_mixed_radix_const256<FFT_MIN + Pows>...};
  return dispatch[n_pow - FFT_MIN](vec);
}
uint32_t reverse_mixed_radix_dyn(int n_pow, uint32_t number) {
  auto res = reverse_mixed_radix_dyn(
      n_pow,
      _mm256_castsi128_si256(_mm_cvtsi32_si128(static_cast<int>(number))),
      std::make_integer_sequence<int, FFT_MAX - FFT_MIN + 1>());
  return static_cast<uint32_t>(_mm256_cvtsi256_si32(res));
}
std::array<uint32_t, 8> reverse_mixed_radix_dyn(int n_pow, uint32_t a,
                                                uint32_t b, uint32_t c,
                                                uint32_t d, uint32_t e,
                                                uint32_t f, uint32_t g,
                                                uint32_t h) {
  auto res = reverse_mixed_radix_dyn(
      n_pow,
      _mm256_set_epi32(static_cast<int>(h), static_cast<int>(g),
                       static_cast<int>(f), static_cast<int>(e),
                       static_cast<int>(d), static_cast<int>(c),
                       static_cast<int>(b), static_cast<int>(a)),
      std::make_integer_sequence<int, FFT_MAX - FFT_MIN + 1>());
  return {static_cast<uint32_t>(_mm256_extract_epi32(res, 0)),
          static_cast<uint32_t>(_mm256_extract_epi32(res, 1)),
          static_cast<uint32_t>(_mm256_extract_epi32(res, 2)),
          static_cast<uint32_t>(_mm256_extract_epi32(res, 3)),
          static_cast<uint32_t>(_mm256_extract_epi32(res, 4)),
          static_cast<uint32_t>(_mm256_extract_epi32(res, 5)),
          static_cast<uint32_t>(_mm256_extract_epi32(res, 6)),
          static_cast<uint32_t>(_mm256_extract_epi32(res, 7))};
}
std::array<uint32_t, 4> reverse_mixed_radix_dyn(int n_pow, uint32_t a,
                                                uint32_t b, uint32_t c,
                                                uint32_t d) {
  auto res = reverse_mixed_radix_dyn(
      n_pow,
      _mm256_castsi128_si256(
          _mm_set_epi32(static_cast<int>(d), static_cast<int>(c),
                        static_cast<int>(b), static_cast<int>(a))),
      std::make_integer_sequence<int, FFT_MAX - FFT_MIN + 1>());
  return {static_cast<uint32_t>(_mm256_extract_epi32(res, 0)),
          static_cast<uint32_t>(_mm256_extract_epi32(res, 1)),
          static_cast<uint32_t>(_mm256_extract_epi32(res, 2)),
          static_cast<uint32_t>(_mm256_extract_epi32(res, 3))};
}

void ensure_twiddle_factors(int want_n_pow) {
  static constexpr double PI = 3.1415926535897931;
  while (cosines_n_pow < want_n_pow) {
    cosines_n_pow++;
    size_t n = size_t{1} << cosines_n_pow;
    std::unique_ptr<double[]> new_cosines{new double[n * 2]};
    memcpy64(reinterpret_cast<uint64_t *>(new_cosines.get()),
             reinterpret_cast<uint64_t *>(cosines.get()), n);
    cosines = std::move(new_cosines);
    double coeff = 2 * PI / static_cast<double>(n);
    for (size_t k = 0; k < n / 2; k++) {
      double c = std::cos(coeff * static_cast<double>(k));
      cosines[n + k] = c;
      cosines[n + n / 2 + k] = -c;
    }
  }
}

auto fft_cooley_tukey(Complex *data, int n_pow) {
  ensure_twiddle_factors(n_pow);
  int count3 = get_counts(n_pow).first;
  fft_cooley_tukey_no_transpose_8(data, n_pow, count3);
  return [n_pow](auto... args) {
    return reverse_mixed_radix_dyn(n_pow, static_cast<uint32_t>(args)...);
  };
}

// Credits to https://stackoverflow.com/a/41148578
// Only work for inputs in the range: [0, 2^52)
__m256i double_to_uint64(__m256d x) {
  x = _mm256_add_pd(x, _mm256_set1_pd(0x0010000000000000));
  return _mm256_castpd_si256(
      _mm256_xor_pd(x, _mm256_set1_pd(0x0010000000000000)));
}
__m256d uint64_to_double(__m256i x) {
  auto y = _mm256_castsi256_pd(x);
  y = _mm256_or_pd(y, _mm256_set1_pd(0x0010000000000000));
  return _mm256_sub_pd(y, _mm256_set1_pd(0x0010000000000000));
}

int get_fft_n_pow(ConstRef lhs, ConstRef rhs) {
  return 64 - __builtin_clzll((lhs.data.size() + rhs.data.size() - 1) * 4 - 1);
}

BigInt mul_fft(ConstRef lhs, ConstRef rhs) {
  int n_pow = get_fft_n_pow(lhs, rhs);
  size_t n = size_t{1} << n_pow;

  // Split numbers into words
  Complex *united_fft = new (std::align_val_t(32)) Complex[n];
  memzero64(reinterpret_cast<uint64_t *>(united_fft), 2 * n);

  const uint16_t *lhs_data =
      reinterpret_cast<const uint16_t *>(lhs.data.data());
  for (size_t i = 0; i < lhs.data.size() * 4; i++) {
    united_fft[i][0] = lhs_data[i];
  }
  const uint16_t *rhs_data =
      reinterpret_cast<const uint16_t *>(rhs.data.data());
  for (size_t i = 0; i < rhs.data.size() * 4; i++) {
    united_fft[i][1] = rhs_data[i];
  }

  // Parallel FFT for lhs and rhs
  auto united_fft_access = fft_cooley_tukey(united_fft, n_pow);

  // Disentangle real and imaginary values into values of lhs & rhs at roots
  // of unity, and then compute FFT of the product as pointwise product of
  // values of lhs and rhs at roots of unity
  size_t united_fft_one = united_fft_access(1);
  auto get_long_fft_times4 = [&](size_t ai, size_t ai4, size_t ani0,
                                 size_t ani1, size_t ani04, size_t ani14) {
    __builtin_prefetch(united_fft[ai4]);
    __builtin_prefetch(united_fft[ai4 | united_fft_one]);
    __builtin_prefetch(united_fft[ani04]);
    __builtin_prefetch(united_fft[ani14]);
    __m128d z0 = _mm_load_pd(united_fft[ai]);
    __m128d z1 = _mm_load_pd(united_fft[ai | united_fft_one]);
    __m256d z01 = _mm256_set_m128d(z1, z0);
    __m128d nz0 = _mm_load_pd(united_fft[ani0]);
    __m128d nz1 = _mm_load_pd(united_fft[ani1]);
    __m256d nz01 = _mm256_set_m128d(nz1, nz0);
    __m256d a = _mm256_add_pd(z01, nz01);
    __m256d b = _mm256_sub_pd(z01, nz01);
    __m256d c = _mm256_blend_pd(a, b, 10);
    __m256d d = _mm256_permute_pd(a, 15);
    __m256d g = _mm256_mul_pd(_mm256_permute_pd(c, 5), _mm256_movedup_pd(b));
    return _mm256_fmsubadd_pd(c, d, g);
  };

  // Treating long_fft as FFT(p(x^2) + x q(x^2)), convert it to FFT(p(x) + i
  // q(x)) by using the fact that p(x) and q(x) have real coefficients, so
  // that we only perform half the work
  Complex *short_fft = new (std::align_val_t(32)) Complex[n / 2];
  for (size_t i = 0; i < n / 2; i += 2) {
    size_t ni0a = i == 0 ? 0 : n - i;
    size_t ni1a = n - i - 1;
    size_t ni0b = n / 2 - i;
    size_t ni1b = n / 2 - i - 1;
    auto [aia, aia4, aib, aib4] =
        united_fft_access(i, i + 4, n / 2 + i, n / 2 + i + 4);
    auto [ani0a, ani1a, ani0a4, ani1a4, ani0b, ani1b, ani0b4, ani1b4] =
        united_fft_access(ni0a, ni1a, ni0a - 4, ni1a - 4, ni0b, ni1b, ni0b - 4,
                          ni1b - 4);
    __builtin_prefetch(&cosines[n + n / 4 + i + 6]);
    __builtin_prefetch(&cosines[n + i + 6]);
    __m256d a = get_long_fft_times4(aia, aia4, ani0a, ani1a, ani0a4, ani1a4);
    __m256d b = get_long_fft_times4(aib, aib4, ani0b, ani1b, ani0b4, ani1b4);
    __m128d w_real01 = _mm_load_pd(&cosines[n + n / 4 + i]);
    __m256d w_real0011 =
        _mm256_permute4x64_pd(_mm256_castpd128_pd256(w_real01), 0x50);
    __m128d w_imag01 = _mm_load_pd(&cosines[n + i]);
    __m256d w_imag0011 =
        _mm256_permute4x64_pd(_mm256_castpd128_pd256(w_imag01), 0x50);
    __m256d c = _mm256_add_pd(a, b);
    __m256d d = _mm256_sub_pd(a, b);
    __m256d e = _mm256_permute_pd(d, 5);
    __m256d f =
        _mm256_fmaddsub_pd(w_real0011, d, _mm256_fmaddsub_pd(w_imag0011, e, c));
    __m256d g = _mm256_mul_pd(f, _mm256_set1_pd(0.125));
    _mm256_store_pd(short_fft[i], g);
  }

  ::operator delete[](united_fft, std::align_val_t(32));

  auto short_fft_access = fft_cooley_tukey(short_fft, n_pow - 1);

  BigInt result;
  result.data.increase_size(size_t{1} << (n_pow - 2));

  uint64_t carry = 0;
  __m256d max_error_vec = _mm256_setzero_pd();

  for (size_t i = 0; i < n / 2; i += 2) {
    size_t ni0 = i == 0 ? 0 : n / 2 - i;
    size_t ni1 = n / 2 - i - 1;
    auto [ani0, ani1, ani08, ani18] =
        short_fft_access(ni0, ni1, ni0 - 8, ni1 - 8);

    __builtin_prefetch(short_fft[ani08]);
    __builtin_prefetch(short_fft[ani18]);

    __m128d z0 = _mm_load_pd(short_fft[ani0]);
    __m128d z1 = _mm_load_pd(short_fft[ani1]);
    __m256d z01 = _mm256_set_m128d(z1, z0);

    __m256d fp_value =
        _mm256_mul_pd(z01, _mm256_set1_pd(2. / static_cast<double>(n)));
    __m256i value = double_to_uint64(fp_value);
    __m256d error = _mm256_andnot_pd(
        _mm256_set1_pd(-0.), _mm256_sub_pd(fp_value, uint64_to_double(value)));
    max_error_vec = _mm256_max_pd(max_error_vec, error);
    __uint128_t tmp = (carry + static_cast<uint64_t>(value[0]) +
                       ((static_cast<uint64_t>(value[1]) +
                         ((static_cast<uint64_t>(value[2]) +
                           (static_cast<__uint128_t>(value[3]) << 16))
                          << 16))
                        << 16));
    result.data[i / 2] = static_cast<uint64_t>(tmp);
    carry = static_cast<uint64_t>(tmp >> 64);
  }

  ::operator delete[](short_fft, std::align_val_t(32));

  __m128d max_error_vec128 =
      _mm_max_pd(_mm256_castpd256_pd128(max_error_vec),
                 _mm256_extractf128_pd(max_error_vec, 1));
  double max_error = std::max(_mm_cvtsd_f64(max_error_vec128),
                              _mm_cvtsd_f64(_mm_castsi128_pd(_mm_srli_si128(
                                  _mm_castpd_si128(max_error_vec128), 8))));

  // if (max_error >= 0.05) {
  //   std::cerr << max_error << " " << n_pow << std::endl;
  // }
  ensure(max_error < 0.4);

  if (carry > 0) {
    result.data.push_back(carry);
  } else {
    result._normalize_nonzero();
  }

  // std::cerr << lhs.data.size() + rhs.data.size() << " -> " <<
  // result.data.size() << std::endl;
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

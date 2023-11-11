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

using Complex = double[2];

inline auto twiddles = new (std::align_val_t(32)) Complex[1];
inline int twiddles_n_pow = -1;

static const __m256d mul_by_j_constant = _mm256_set_pd(-0., 0., -0., 0.);

// j = -i
// Take pressure off p5 by using integer operations
__m256d mul_by_j_reg(__m256d vec) {
  __m256d result;
  asm volatile(
    "vpxor %[a], %[b], %[c];"
    : [c] "=x"(result)
    : [a] "x"(_mm256_set_pd(-0., 0., -0., 0.)), [b] "x"(_mm256_permute_pd(vec, 5))
  );
  return result;
}
__m256d mul_by_j_mem(__m256d vec) {
  __m256d result;
  asm volatile(
    "vpxor %[a], %[b], %[c];"
    : [c] "=x"(result)
    : [a] "m"(mul_by_j_constant), [b] "x"(_mm256_permute_pd(vec, 5))
  );
  return result;
}

__m256d mul(__m256d a, __m256d b) {
  return _mm256_fmaddsub_pd(
      _mm256_movedup_pd(a), b,
      _mm256_mul_pd(_mm256_permute_pd(a, 15), _mm256_permute_pd(b, 5)));
}

void fft_cooley_tukey_no_transpose_4(Complex *data, int n_pow) {
  size_t old_n = size_t{1} << n_pow;

  while (n_pow > 2) {
    size_t n = size_t{1} << n_pow;
    size_t n2 = size_t{1} << (n_pow - 2);

    for (Complex *cur_data = data; cur_data != data + old_n; cur_data += n) {
      __m256d a0 = _mm256_load_pd(cur_data[0]);
      __m256d a1 = _mm256_load_pd(cur_data[n2]);
      __m256d a2 = _mm256_load_pd(cur_data[n2 * 2]);
      __m256d a3 = _mm256_load_pd(cur_data[n2 * 3]);

      for (size_t i = 0; i < n2; i += 2) {
        __m256d next_a0 = _mm256_load_pd(cur_data[i + 2]);
        __m256d next_a1 = _mm256_load_pd(cur_data[n2 + i + 2]);
        __m256d next_a2 = _mm256_load_pd(cur_data[n2 * 2 + i + 2]);
        __m256d next_a3 = _mm256_load_pd(cur_data[n2 * 3 + i + 2]);

        __m256d c0 = _mm256_add_pd(a0, a2);
        __m256d c1 = _mm256_add_pd(a1, a3);
        __m256d c2 = _mm256_sub_pd(a0, a2);
        __m256d c3 = mul_by_j_reg(_mm256_sub_pd(a1, a3));

        __m256d b0 = _mm256_add_pd(c0, c1);
        __m256d b1 = _mm256_add_pd(c2, c3);
        __m256d b2 = _mm256_sub_pd(c0, c1);
        __m256d b3 = _mm256_sub_pd(c2, c3);

        __m256d w1 = _mm256_load_pd(twiddles[n + i]);
        __m256d w2 = _mm256_load_pd(twiddles[n / 2 + i]);
        __m256d w3 = mul(w1, w2);

        _mm256_store_pd(cur_data[i], b0);
        _mm256_store_pd(cur_data[n2 + i], mul(w1, b1));
        _mm256_store_pd(cur_data[n2 * 2 + i], mul(w2, b2));
        _mm256_store_pd(cur_data[n2 * 3 + i], mul(w3, b3));

        a0 = next_a0;
        a1 = next_a1;
        a2 = next_a2;
        a3 = next_a3;
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
    __m256d f1 = mul_by_j_mem(_mm256_sub_pd(a2, a6));

    __m256d c0 = _mm256_add_pd(e0, f0);
    __m256d c1 = _mm256_add_pd(e1, f1);
    __m256d c2 = _mm256_sub_pd(e0, f0);
    __m256d c3 = _mm256_sub_pd(e1, f1);

    __m256d g0 = _mm256_add_pd(a1, a5);
    __m256d g1 = _mm256_sub_pd(a1, a5);

    __m256d h0 = _mm256_add_pd(a3, a7);
    __m256d h1 = mul_by_j_mem(_mm256_sub_pd(a3, a7));

    __m256d k0 = _mm256_mul_pd(_mm256_add_pd(g1, h1), rsqrt2);
    __m256d k1 = _mm256_mul_pd(_mm256_sub_pd(g1, h1), rsqrt2);

    __m256d d0 = _mm256_add_pd(g0, h0);
    __m256d d1 = _mm256_add_pd(mul_by_j_mem(k0), k0);
    __m256d d2 = mul_by_j_mem(_mm256_sub_pd(g0, h0));
    __m256d d3 = _mm256_sub_pd(mul_by_j_mem(k1), k1);

    __m256d b0 = _mm256_add_pd(c0, d0);
    __m256d b1 = _mm256_add_pd(c1, d1);
    __m256d b2 = _mm256_add_pd(c2, d2);
    __m256d b3 = _mm256_add_pd(c3, d3);
    __m256d b4 = _mm256_sub_pd(c0, d0);
    __m256d b5 = _mm256_sub_pd(c1, d1);
    __m256d b6 = _mm256_sub_pd(c2, d2);
    __m256d b7 = _mm256_sub_pd(c3, d3);

    _mm256_store_pd(data[i], b0);
    __m256d w1 = _mm256_load_pd(twiddles[n + i]);
    _mm256_store_pd(data[n2 + i], mul(w1, b1));
    __m256d w2 = _mm256_load_pd(twiddles[n / 2 + i]);
    _mm256_store_pd(data[n2 * 2 + i], mul(w2, b2));
    __m256d w3 = mul(w1, w2);
    _mm256_store_pd(data[n2 * 3 + i], mul(w3, b3));
    __m256d w4 = _mm256_load_pd(twiddles[n / 4 + i]);
    _mm256_store_pd(data[n2 * 4 + i], mul(w4, b4));
    __m256d w5 = mul(w1, w4);
    _mm256_store_pd(data[n2 * 5 + i], mul(w5, b5));
    __m256d w6 = mul(w3, w3);
    _mm256_store_pd(data[n2 * 6 + i], mul(w6, b6));
    __m256d w7 = mul(w4, w3);
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
reverse_mixed_radix_impl(__m256i number,
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
template <int N_POW>
__attribute__((always_inline)) inline __m256i reverse_mixed_radix(__m256i number) {
  static constexpr int COUNT3 = get_counts(N_POW).first;
  static constexpr int COUNT2 = get_counts(N_POW).second;
  return reverse_mixed_radix_impl<N_POW>(
      number, std::make_integer_sequence<int, COUNT3>(),
      std::make_integer_sequence<int, COUNT2>());
}

void ensure_twiddle_factors(int want_n_pow) {
  static constexpr double PI = 3.1415926535897931;
  while (twiddles_n_pow < want_n_pow) {
    twiddles_n_pow++;
    size_t n = size_t{1} << twiddles_n_pow;
    auto new_twiddles = new (std::align_val_t(32)) Complex[n * 2];
    memcpy64(reinterpret_cast<uint64_t *>(new_twiddles),
             reinterpret_cast<uint64_t *>(twiddles), n * 2);
    ::operator delete[](twiddles, std::align_val_t(32));
    twiddles = new_twiddles;
    double coeff = 2 * PI / static_cast<double>(n);
    for (size_t k = 0; k < n / 2; k++) {
      double c = std::cos(coeff * static_cast<double>(k));
      twiddles[n + k][0] = c;
      twiddles[n + n / 4 + k][1] = -c;
      twiddles[n + n / 2 + k][0] = -c;
      size_t i = n / 4 * 3 + k;
      if (i >= n) {
        i -= n;
      }
      twiddles[n + i][1] = c;
    }
  }
}

void fft_cooley_tukey(Complex *data, int n_pow) {
  ensure_twiddle_factors(n_pow);
  int count3 = get_counts(n_pow).first;
  fft_cooley_tukey_no_transpose_8(data, n_pow, count3);
}

// Credits to https://stackoverflow.com/a/41148578
// Only work for inputs in the range: [-0.25, 2^52 + 0.5)
__m256d uint64_to_double(__m256i x) {
  auto y = _mm256_castsi256_pd(x);
  y = _mm256_or_pd(y, _mm256_set1_pd(0x0010000000000000));
  return _mm256_sub_pd(y, _mm256_set1_pd(0x0010000000000000));
}

int get_fft_n_pow(ConstRef lhs, ConstRef rhs) {
  return 64 - __builtin_clzll((lhs.data.size() + rhs.data.size() - 1) * 4 - 1);
}

void fft_dif() {

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

  fft_dif(input_fft, n_pow);

  return input_fft;
}

Complex* mul_fft_middle_end(Complex *lhs_fft_dif, Complex *rhs_fft_dif, int n_pow) {
  size_t n = size_t{1} << n_pow;

  // CT4 reads out of bounds, so add + 2
  Complex *prod_fft_dif = new (std::align_val_t(32)) Complex[n / 2 + 2];

  auto handle_iteration = [&](size_t k, size_t k_complement) {
    Complex lhs_k = lhs_fft_dif[k];
    Complex rhs_k = rhs_fft_dif[k];
    Complex lhs_k_complement = lhs_fft_dif[k_complement];
    Complex rhs_k_complement = rhs_fft_dif[k_complement];
    Complex one_plus_twiddles_div_4_k = one_plus_twiddles_div_4_bitreversed[k];
    Complex result_k = prod_fft_dif[k];
    // lhs_k * rhs_k - one_plus_twiddles_div_4_bitreversed[k] * (lhs_k - lhs_k_complement.conj()) * (rhs_k - rhs_k_complement.conj())

    double a_real = lhs_k[0] - lhs_k_complement[0];
    double a_imag = lhs_k[1] + lhs_k_complement[1];
    double b_real = rhs_k[0] - rhs_k_complement[0];
    double b_imag = rhs_k[1] + rhs_k_complement[1];
    double c_real = a_real * b_real - a_imag * b_imag;
    double c_imag = a_real * b_imag + a_imag * b_real;

    result_k[0] = (
      (lhs_k[0] * rhs_k[0] - lhs_k[1] * rhs_k[1])
      - (one_plus_twiddles_div_4_k[0] * c_real - one_plus_twiddles_div_4_k[1] * c_imag)
    );
    result_k[1] = (
      (lhs_k[0] * rhs_k[1] + lhs_k[1] * rhs_k[0])
      - (one_plus_twiddles_div_4_k[0] * c_imag + one_plus_twiddles_div_4_k[1] * c_real)
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

Complex* mul_fft_transform_output(Complex *prod_fft_dif, int n_pow) {
  size_t n = size_t{1} << n_pow;

  fft_dit(prod_fft_dif, n_pow - 1);

  for (size_t k = 0; k < n / 2; k++) {
    size_t k1 = k == 0 ? 0 : n / 2 - k;
    prod_fft_dif[k1][0] * (static_cast<double>(2) / n)
    prod_fft_dif[k1][1] * (static_cast<double>(2) / n)
  }

  //     Pr_{2k}   = Re Qr_k = Re Qr_{-k} * 2/n
  //     Pr_{2k+1} = Im Qr_k = Im Qr_{-k} * 2/n

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

  Complex *lhs_fft_dif = mul_fft_transform_input(lhs, n_pow);
  Complex *rhs_fft_dif = mul_fft_transform_input(rhs, n_pow);

  Complex *prod_fft_dif = mul_fft_middle_end(lhs_fft_dif, rhs_fft_dif, n_pow);

  ::operator delete[](lhs_fft_dif, std::align_val_t(32));
  ::operator delete[](rhs_fft_dif, std::align_val_t(32));

  mul_fft_transform_output(prod_fft_dif, n_pow);


  // const size_t n = size_t{1} << N_POW;

  // // Split numbers into words
  // // CT4 reads out of bounds; overaligning to a cache line is more efficient wrt. prefetch
  // Complex *united_fft = new (std::align_val_t(64)) Complex[n + 4];
  // const uint16_t *lhs_data = reinterpret_cast<const uint16_t *>(lhs.data.data());
  // const uint16_t *rhs_data = reinterpret_cast<const uint16_t *>(rhs.data.data());
  // for (size_t i = 0; i < lhs.data.size(); i++) {
  //   __m128i a = _mm_unpacklo_epi16( _mm_cvtsi64_si128(lhs.data[i]), _mm_cvtsi64_si128(rhs.data[i]));
  //   __m256d b = uint64_to_double(_mm256_cvtepu16_epi64(a));
  //   __m256d c = uint64_to_double(_mm256_cvtepu16_epi64(_mm_castpd_si128(_mm_permute_pd(_mm_castsi128_pd(a), 1))));
  //   _mm256_store_pd(united_fft[i * 4], b);
  //   _mm256_store_pd(united_fft[i * 4 + 2], c);
  // }
  // for (size_t i = lhs.data.size(); i < rhs.data.size(); i++) {
  //   __m256d a = uint64_to_double(_mm256_cvtepu16_epi64(_mm_shufflelo_epi16(_mm_cvtsi64_si128(rhs.data[i]), 0b11011000)));
  //   __m256d b = _mm256_unpacklo_pd(_mm256_setzero_pd(), a);
  //   __m256d c = _mm256_blend_pd(_mm256_setzero_pd(), a, 10);
  //   _mm256_store_pd(united_fft[i * 4], b);
  //   _mm256_store_pd(united_fft[i * 4 + 2], c);
  // }
  // memzero64(reinterpret_cast<uint64_t*>(united_fft + rhs.data.size() * 4), 2 * (n + 4 - rhs.data.size() * 4));

  // // Parallel FFT for lhs and rhs
  // fft_cooley_tukey(united_fft, N_POW);

  // // Treating long_fft as FFT(p(x^2) + x q(x^2)), convert it to FFT(p(x) + i
  // // q(x)) by using the fact that p(x) and q(x) have real coefficients, so
  // // that we only perform half the work
  // Complex *short_fft = new (std::align_val_t(32)) Complex[n / 2 + 4];  // CT4 reads out of bounds

  // // Disentangle real and imaginary values into values of lhs & rhs at roots
  // // of unity, and then compute FFT of the product as pointwise product of
  // // values of lhs and rhs at roots of unity

  // auto r = reverse_mixed_radix<N_POW>(_mm256_set_epi32(0, 0, 0, 0, 0, 0, n / 2, 1));
  // const size_t united_fft_one = static_cast<uint32_t>(_mm256_extract_epi32(r, 0));
  // const size_t united_fft_half_n = static_cast<uint32_t>(_mm256_extract_epi32(r, 1));

  // auto get_long_fft_times4 = [&](size_t ai0, size_t ai1, size_t ani0, size_t ani1) {
  //   __m128d z0 = _mm_load_pd(united_fft[ai0]);
  //   __m128d z1 = _mm_load_pd(united_fft[ai1]);
  //   __m256d z01 = _mm256_set_m128d(z1, z0);
  //   __m128d nz0 = _mm_load_pd(united_fft[ani0]);
  //   __m128d nz1 = _mm_load_pd(united_fft[ani1]);
  //   __m256d nz01 = _mm256_set_m128d(nz1, nz0);
  //   __m256d a = _mm256_add_pd(z01, nz01);
  //   __m256d b = _mm256_sub_pd(z01, nz01);
  //   __m256d c = _mm256_blend_pd(a, b, 10);
  //   __m256d d = _mm256_permute_pd(a, 15);
  //   __m256d g = _mm256_mul_pd(_mm256_permute_pd(c, 5), _mm256_movedup_pd(b));
  //   return _mm256_fmsubadd_pd(c, d, g);
  // };

  // auto twiddles_cur = twiddles + n;
  // // As we have free slots, spend them for prefetch
  // __m256i query = _mm256_set_epi32(0, n / 2 - 4 - 1, n / 2 - 4, 4, 0, n / 2 - 1, n / 2, 0);
  // for (size_t i = 0; i < n / 2; i += 2) {
  //   auto r = reverse_mixed_radix<N_POW>(query);

  //   auto ai0a = static_cast<uint32_t>(_mm256_extract_epi32(r, 0));
  //   auto ani0b = static_cast<uint32_t>(_mm256_extract_epi32(r, 1));
  //   auto ani1b = static_cast<uint32_t>(_mm256_extract_epi32(r, 2));
  //   auto ai1a = ai0a + united_fft_one;
  //   auto ai0b = ai0a + united_fft_half_n;
  //   auto ai1b = ai0b + united_fft_one;
  //   auto ani0a = i == 0 ? 0 : ani0b + united_fft_half_n;
  //   auto ani1a = ani1b + united_fft_half_n;

  //   auto ai0a_prefetch = static_cast<uint32_t>(_mm256_extract_epi32(r, 4));
  //   auto ani0b_prefetch = static_cast<uint32_t>(_mm256_extract_epi32(r, 5));
  //   auto ani1b_prefetch = static_cast<uint32_t>(_mm256_extract_epi32(r, 6));
  //   auto ai1a_prefetch = ai0a_prefetch + united_fft_one;
  //   // ai0b_prefetch is in the same cache line as ai0a_prefetch
  //   auto ai1b_prefetch = ai0a_prefetch + united_fft_half_n + united_fft_one;
  //   // although these are close, they are in a different cache line
  //   auto ani0a_prefetch = ani0b_prefetch + united_fft_half_n;
  //   auto ani1a_prefetch = ani1b_prefetch + united_fft_half_n;

  //   __builtin_prefetch(united_fft[ai0a_prefetch]);
  //   __builtin_prefetch(united_fft[ai1a_prefetch]);
  //   __builtin_prefetch(united_fft[ani0a_prefetch]);
  //   __builtin_prefetch(united_fft[ani1a_prefetch]);
  //   __builtin_prefetch(united_fft[ai1b_prefetch]);
  //   __builtin_prefetch(united_fft[ani0b_prefetch]);
  //   __builtin_prefetch(united_fft[ani1b_prefetch]);
  //   __builtin_prefetch(twiddles_cur[i + 4]);  // I have no idea why, but this significantly reduces cache misses

  //   query = _mm256_add_epi32(query, _mm256_set_epi32(0, -2, -2, 2, 0, -2, -2, 2));

  //   __m256d a = get_long_fft_times4(ai0a, ai1a, ani0a, ani1a);
  //   __m256d b = get_long_fft_times4(ai0b, ai1b, ani0b, ani1b);
  //   __m256d c = _mm256_add_pd(a, b);
  //   __m256d d = _mm256_sub_pd(a, b);
  //   __m256d e = _mm256_permute_pd(d, 5);
  //   __m256d w = _mm256_load_pd(twiddles_cur[i]);
  //   __m256d w0 = _mm256_movedup_pd(w);
  //   __m256d w1 = _mm256_permute_pd(w, 15);
  //   __m256d f = _mm256_fmaddsub_pd(w1, d, _mm256_fmaddsub_pd(w0, e, c));
  //   __m256d g = _mm256_mul_pd(f, _mm256_set1_pd(0.125));
  //   _mm256_store_pd(short_fft[i], g);
  // }

  // ::operator delete[](united_fft, std::align_val_t(64));

  // fft_cooley_tukey(short_fft, N_POW - 1);

  // BigInt result;
  // result.data.increase_size(size_t{1} << (N_POW - 2));

  // uint64_t carry = 0;

  // __m128i query2 = _mm_set_epi32(n / 2 - 4 - 1, n / 2 - 4, n / 2 - 1, n / 2);
  // for (size_t i = 0; i < n / 2; i += 2) {
  //   auto r = reverse_mixed_radix<N_POW - 1>(_mm256_castsi128_si256(query2));
  //   auto ani0 = static_cast<uint32_t>(_mm256_extract_epi32(r, 0));
  //   auto ani1 = static_cast<uint32_t>(_mm256_extract_epi32(r, 1));
  //   auto ani0_prefetch = static_cast<uint32_t>(_mm256_extract_epi32(r, 2));
  //   auto ani1_prefetch = static_cast<uint32_t>(_mm256_extract_epi32(r, 3));
  //   query2 = _mm_sub_epi32(query2, _mm_set1_epi32(2));

  //   __builtin_prefetch(short_fft[ani0_prefetch]);
  //   __builtin_prefetch(short_fft[ani1_prefetch]);

  //   __m128d z0 = _mm_load_pd(short_fft[ani0]);
  //   __m128d z1 = _mm_load_pd(short_fft[ani1]);
  //   __m256d z01 = _mm256_set_m128d(z1, z0);

  //   // Convert z01 to integer, dividing it by n/2 in the process
  //   __m256d shift_const = _mm256_set1_pd(static_cast<double>(0x0010000000000000) * (n / 2));
  //   __m256i value = _mm256_castpd_si256(_mm256_xor_pd(_mm256_add_pd(z01, shift_const), shift_const));

  //   __uint128_t tmp = static_cast<uint64_t>(value[3]);
  //   tmp = (tmp << 16) + static_cast<uint64_t>(value[2]);
  //   tmp = (tmp << 16) + static_cast<uint64_t>(value[1]);
  //   tmp = (tmp << 16) + static_cast<uint64_t>(value[0]);
  //   tmp += carry;
  //   result.data[i / 2] = static_cast<uint64_t>(tmp);
  //   carry = static_cast<uint64_t>(tmp >> 64);
  // }

  // ::operator delete[](short_fft, std::align_val_t(32));

  // if (carry > 0) {
  //   result.data.push_back(carry);
  // } else {
  //   result._normalize_nonzero();
  // }

  // // std::cerr << lhs.data.size() + rhs.data.size() << " -> " <<
  // // result.data.size() << std::endl;
  // ensure(result.data.size() == lhs.data.size() + rhs.data.size() ||
  //        result.data.size() == lhs.data.size() + rhs.data.size() - 1);

  // return result;
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

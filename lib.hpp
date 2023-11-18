#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
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

struct EXPORT with_base {
  uint64_t base;
};

class ConstSpan;

class SmallVec {
  static constexpr size_t INLINE_STORAGE_SIZE = 48;

  uint64_t* _begin;
  size_t _size;
  size_t _capacity;
  uint64_t _inline_storage[INLINE_STORAGE_SIZE];

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
    memcpy(_begin, data, size * 8);
    _size = size;
  }

  SmallVec(ConstSpan rhs);
  SmallVec(const SmallVec& rhs);
  SmallVec(SmallVec&& rhs) {
    if (rhs._begin == rhs._inline_storage) {
      _begin = _inline_storage;
      memcpy(_begin, rhs._begin, rhs._size * 8);
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
      memcpy(_begin, rhs._begin, rhs._size * 8);
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

  void increase_capacity_to(size_t new_capacity) {
    uint64_t* new_begin = new uint64_t[new_capacity];
    memcpy(new_begin, _begin, _size * 8);
    if (_begin != _inline_storage) {
      delete[] _begin;
    }
    _begin = new_begin;
    _capacity = new_capacity;
  }
  void overwrite_zerofill_leaking(size_t size) {
    if (size > INLINE_STORAGE_SIZE) {
      _begin = new uint64_t[size];
      _capacity = size;
    }
    memset(_begin, 0, size * 8);
    _size = size;
  }
  void increase_size_zerofill(size_t new_size) {
    if (_capacity < new_size) {
      increase_capacity_to(new_size);
    }
    memset(_begin + _size, 0, (new_size - _size) * 8);
    _size = new_size;
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
  memcpy(_begin, rhs.data(), rhs.size() * 8);
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
  memcpy(_begin, rhs.data(), rhs.size() * 8);
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

public:
  SmallVec data;

  BigInt();
  BigInt(const uint64_t* data, size_t size);

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

  explicit operator BigInt() const { return {data.data(), data.size()}; }

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

inline constexpr int FFT_RECURSIVE = 10;
inline constexpr int FFT_MAX_16BIT = 18;  // a bit less than 52 - 16 * 2
inline constexpr int FFT_MAX_16BIT_RELIABLE = 16;  // 2 bits below known counter-example
inline constexpr int FFT_MAX_12BIT = 26;  // a bit less than 52 - 12 * 2

struct Complex {
  double real, imag;
  Complex minus_conj(Complex b) const {
    return {real - b.real, imag + b.imag};
  }
};
Complex operator-(Complex a) {
  return {-a.real, -a.imag};
}
Complex operator-(Complex a, Complex b) {
  return {a.real - b.real, a.imag - b.imag};
}
Complex operator*(Complex a, Complex b) {
  return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}
Complex operator*(Complex a, double b) {
  return {a.real * b, a.imag * b};
}
Complex operator+(Complex a, double b) {
  return {a.real + b, a.imag};
}

struct Complex4 {
  __m256d real, imag;
  Complex4(__m256d real, __m256d imag): real(real), imag(imag) {}
  Complex4(Complex value): real(_mm256_set1_pd(value.real)), imag(_mm256_set1_pd(value.imag)) {}
  Complex4 conj_mul(Complex4 b) const {
    return {
      _mm256_fmadd_pd(real, b.real, _mm256_mul_pd(imag, b.imag)),
      _mm256_fmsub_pd(real, b.imag, _mm256_mul_pd(imag, b.real))
    };
  }
  Complex4 minus_conj(Complex4 b) const {
    return {_mm256_sub_pd(real, b.real), _mm256_add_pd(imag, b.imag)};
  }
  // real0, real1, real2, real3 -> real3, real2, real1, real0, same with imag
  Complex4 reverse() const {
    return {_mm256_permute4x64_pd(real, 0x1b), _mm256_permute4x64_pd(imag, 0x1b)};
  }
  // a0, a1, a2, a3 -> a0, -a1, a2, -a3
  Complex4 alternate_signs() const {
    __m256d magic = _mm256_set_pd(-0., 0., -0., 0.);
    return {_mm256_xor_pd(real, magic), _mm256_xor_pd(imag, magic)};
  }
  // a * b + c
  Complex4 fmadd(Complex4 b, Complex4 c) const {
    return {
      _mm256_fmsub_pd(real, b.real, _mm256_fmsub_pd(imag, b.imag, c.real)),
      _mm256_fmadd_pd(real, b.imag, _mm256_fmadd_pd(imag, b.real, c.imag))
    };
  }
  // a * b - c
  Complex4 fmsub(Complex4 b, Complex4 c) const {
    return {
      _mm256_fmsub_pd(real, b.real, _mm256_fmadd_pd(imag, b.imag, c.real)),
      _mm256_fmadd_pd(real, b.imag, _mm256_fmsub_pd(imag, b.real, c.imag))
    };
  }
};
Complex4 operator+(Complex4 a, Complex4 b) {
  return {_mm256_add_pd(a.real, b.real), _mm256_add_pd(a.imag, b.imag)};
}
Complex4 operator-(Complex4 a, Complex4 b) {
  return {_mm256_sub_pd(a.real, b.real), _mm256_sub_pd(a.imag, b.imag)};
}
Complex4 operator*(Complex4 a, Complex4 b) {
  return {
    _mm256_fmsub_pd(a.real, b.real, _mm256_mul_pd(a.imag, b.imag)),
    _mm256_fmadd_pd(a.real, b.imag, _mm256_mul_pd(a.imag, b.real))
  };
}
Complex4 operator*(Complex4 a, double b) {
  __m256d bvec = _mm256_set1_pd(b);
  return {_mm256_mul_pd(a.real, bvec), _mm256_mul_pd(a.imag, bvec)};
}

struct Complex2 {
  __m128d real, imag;
  // real0, real1, imag0, imag1
  static Complex2 from_continuous(__m256d value) {
    return {_mm256_castpd256_pd128(value), _mm256_extractf128_pd(value, 1)};
  }
  // real0, imag0, real1, imag1
  __m256d to_interleaved() const {
    return _mm256_set_m128d(_mm_unpackhi_pd(real, imag), _mm_unpacklo_pd(real, imag));
  }
  // real0, real1 -> real0, real0, real1, real1, same with imag
  Complex4 duplicate_continuous() const {
    return {
      _mm256_permute4x64_pd(_mm256_castpd128_pd256(real), 0x50),
      _mm256_permute4x64_pd(_mm256_castpd128_pd256(imag), 0x50)
    };
  }
};

std::array<__m256d, 4> transpose(__m256d a0, __m256d a1, __m256d a2, __m256d a3) {
  __m256d b0 = _mm256_unpacklo_pd(a0, a1);
  __m256d b1 = _mm256_unpackhi_pd(a0, a1);
  __m256d b2 = _mm256_unpacklo_pd(a2, a3);
  __m256d b3 = _mm256_unpackhi_pd(a2, a3);
  return {_mm256_permute2f128_pd(b0, b2, 0x20), _mm256_permute2f128_pd(b1, b3, 0x20),
          _mm256_permute2f128_pd(b0, b2, 0x31), _mm256_permute2f128_pd(b1, b3, 0x31)};
}

struct ComplexSpan {
  // Interleaved storage: real0..3, imag0..3, real4..7, imag4..7, ...
  double* data;
  ComplexSpan(double* data) : data(data) {}

  Complex read1(size_t i) const {
    i = i / 4 * 8 + i % 4;
    return {data[i], data[i + 4]};
  }
  void write1(size_t i, Complex value) {
    i = i / 4 * 8 + i % 4;
    data[i] = value.real;
    data[i + 4] = value.imag;
  }

  // Data stored as real0, real1, ..., imag0, imag1
  Complex2 read2(size_t i) const {
    i = i / 4 * 8 + i % 4;
    return {_mm_load_pd(&data[i]), _mm_load_pd(&data[i + 4])};
  }
  void write2(size_t i, Complex2 value) {
    i = i / 4 * 8 + i % 4;
    _mm_store_pd(&data[i], value.real);
    _mm_store_pd(&data[i + 4], value.imag);
  }

  // Data stored as real0, real1, real2, real3, ..., imag0, imag1, imag2, imag3
  Complex4 read4(size_t i) const {
    return {_mm256_load_pd(&data[i * 2]), _mm256_load_pd(&data[i * 2 + 4])};
  }
  void write4(size_t i, Complex4 value) {
    _mm256_store_pd(&data[i * 2], value.real);
    _mm256_store_pd(&data[i * 2 + 4], value.imag);
  }

  // Data stored as real0, real4, real8, real12, real1, real5, real9, real13, real2, real6, real10,
  // real14, real3, real7, real11, real15, ..., <same with imag>. This pattern occurs in low levels
  // of FFT
  std::array<Complex4, 4> read_transposed44(size_t i) const {
    auto a0 = read4(i);
    auto a1 = read4(i + 4);
    auto a2 = read4(i + 8);
    auto a3 = read4(i + 12);
    auto [b0_real, b1_real, b2_real, b3_real] = transpose(a0.real, a1.real, a2.real, a3.real);
    auto [b0_imag, b1_imag, b2_imag, b3_imag] = transpose(a0.imag, a1.imag, a2.imag, a3.imag);
    return {{
      {b0_real, b0_imag},
      {b1_real, b1_imag},
      {b2_real, b2_imag},
      {b3_real, b3_imag}
    }};
  }
  void write_transposed44(size_t i, Complex4 a0, Complex4 a1, Complex4 a2, Complex4 a3) {
    auto [b0_real, b1_real, b2_real, b3_real] = transpose(a0.real, a1.real, a2.real, a3.real);
    auto [b0_imag, b1_imag, b2_imag, b3_imag] = transpose(a0.imag, a1.imag, a2.imag, a3.imag);
    write4(i, {b0_real, b0_imag});
    write4(i + 4, {b1_real, b1_imag});
    write4(i + 8, {b2_real, b2_imag});
    write4(i + 12, {b3_real, b3_imag});
  }

  // Data stored as real0, real4, real1, real5, real2, real6, real3, real7, ..., <same with imag>.
  // This pattern arises on vectorizing reads at 2k and 2k+1
  std::array<Complex4, 2> read_transposed24(size_t i) const {
    Complex4 low = read4(i);
    Complex4 high = read4(i + 4);
    return {
      Complex4{
        _mm256_permute4x64_pd(_mm256_unpacklo_pd(low.real, high.real), 0xd8),
        _mm256_permute4x64_pd(_mm256_unpacklo_pd(low.imag, high.imag), 0xd8)
      },
      Complex4{
        _mm256_permute4x64_pd(_mm256_unpackhi_pd(low.real, high.real), 0xd8),
        _mm256_permute4x64_pd(_mm256_unpackhi_pd(low.imag, high.imag), 0xd8)
      }
    };
  }

  void zero_many(size_t i, size_t count) {
    while (i % 4 != 0 && count > 0) {
      write1(i, Complex{0, 0});
      i++;
      count--;
    }
    memset(data + i * 2, 0, count * 16);
  }
};

struct ComplexArray: public ComplexSpan {
  // Overaligning to 64 bytes (as opposed to 32 bytes for SIMD) helps with cache locality because
  // real and imag halves are always in the same cache line
  ComplexArray(size_t n) : ComplexSpan(new (std::align_val_t(64)) double[2 * n]) {}
  ~ComplexArray() {
    ::operator delete[](data, std::align_val_t(64));
  }
  ComplexArray(ComplexArray&& rhs): ComplexSpan(rhs.data) {
    rhs.data = nullptr;
  }
  ComplexArray& operator=(ComplexArray&& rhs) {
    if (this != &rhs) {
      data = rhs.data;
      rhs.data = nullptr;
    }
    return *this;
  }
};

ComplexArray construct_initial_twiddles_bitreversed() {
  // bitreversed k | length | k | n | 2 pi k / n | cos      | sin
  // 0             | 0      | 0 | 2 | 0          | 1        | 0
  // 1             | 1      | 1 | 4 | pi/2       | 0        | 1
  // 2             | 2      | 1 | 8 | pi/4       | 1/sqrt2  | 1/sqrt2
  // 3             | 2      | 3 | 8 | 3pi/4      | -1/sqrt2 | 1/sqrt2
  double r = sqrt(0.5);
  ComplexArray arr(4);
  arr.write1(0, {1, 0});
  arr.write1(1, {0, 1});
  arr.write1(2, {r, r});
  arr.write1(3, {-r, r});
  return arr;
}

inline ComplexArray twiddles_bitreversed = construct_initial_twiddles_bitreversed();
inline std::atomic<int> twiddles_n_pow = 3;
inline std::mutex twiddles_mutex;

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
  if (twiddles_n_pow.load(std::memory_order_acquire) >= want_n_pow) {
    return;
  }

  std::lock_guard lock(twiddles_mutex);

  int cur_twiddles_n_pow = twiddles_n_pow.load(std::memory_order_relaxed);
  if (cur_twiddles_n_pow >= want_n_pow) {
    return;
  }

  static constexpr double PI = 3.1415926535897931;

  // twiddle_bitreversed doesn't depend on n (to be more specific, twiddle_bitreversed of a lesser n
  // is a prefix of twiddle_bitreversed of a larger n)
  size_t old_n = 1uz << cur_twiddles_n_pow;
  size_t new_n = 1uz << want_n_pow;
  ComplexArray new_twiddles_bitreversed(new_n / 2);
  memcpy(new_twiddles_bitreversed.data, twiddles_bitreversed.data, old_n * 8);
  ::operator delete[](twiddles_bitreversed.data, std::align_val_t(64));
  twiddles_bitreversed = std::move(new_twiddles_bitreversed);
  double coeff = 2 * PI / static_cast<double>(new_n);
  for (size_t k = old_n / 4; k < new_n / 4; k++) {
    double angle = coeff * bitreverse(k, want_n_pow - 2);
    double c = std::cos(angle);
    double s = std::sin(angle);
    twiddles_bitreversed.write1(2 * k, {c, s});
    twiddles_bitreversed.write1(2 * k + 1, {-s, c});
  }

  twiddles_n_pow.store(want_n_pow, std::memory_order_relaxed);
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

void fft_dif(ComplexSpan a, int n_pow, size_t k_base) {
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
  //         Complex w = twiddles_bitreversed.read1(k);
  //         for (size_t j = (2 * k) << step; j < ((2 * k + 1) << step); j++) {
  //           Complex e = a.read1(j);
  //           Complex o = a.read1(j + (1uz << step));
  //           Complex wo_real = w * o;
  //           a.write1(j, e + wo);
  //           a.write1(j + (1uz << step), e - wo);
  //         }
  //       }
  //     }

  // Prevent re-reading of twiddles base address after every store, because Clang assumes
  // _mm*_store_* may alias anything
  ComplexSpan twiddles_bitreversed_span(twiddles_bitreversed);
  int step = n_pow - 1;

  // It's critical for performance to perform multiple steps at once
  auto radix4_high = [&]() {
    for (size_t k = k_base; k < k_base + (1uz << (n_pow - 1 - step)); k++) {
      Complex4 w = twiddles_bitreversed_span.read1(k);
      Complex4 w2 = twiddles_bitreversed_span.read1(2 * k);
      Complex4 w3 = twiddles_bitreversed_span.read1(2 * k + 1);

      size_t i1 = 1uz << (step - 1);
      size_t i2 = 2uz << (step - 1);
      size_t i3 = 3uz << (step - 1);
      for (size_t j = (4 * k) << (step - 1); j < ((4 * k + 1) << (step - 1)); j += 4) {
        Complex4 a0 = a.read4(j);
        Complex4 a1 = a.read4(j + i1);
        Complex4 a2 = a.read4(j + i2);
        Complex4 a3 = a.read4(j + i3);
        Complex4 wa2 = w * a2;
        Complex4 wa3 = w * a3;
        Complex4 e2 = a0 + wa2;
        Complex4 e3 = a0 - wa2;
        Complex4 wo2 = w2 * (a1 + wa3);
        Complex4 wo3 = w3 * (a1 - wa3);
        a.write4(j, e2 + wo2);
        a.write4(j + i1, e2 - wo2);
        a.write4(j + i2, e3 + wo3);
        a.write4(j + i3, e3 - wo3);
      }
    }
  };

  auto radix4_low = [&]() {
    ensure(step == 1);

    for (size_t k = k_base; k < k_base + (1uz << (n_pow - 2)); k += 4) {
      Complex4 w = twiddles_bitreversed_span.read4(k);
      auto [w2, w3] = twiddles_bitreversed_span.read_transposed24(2 * k);
      auto [a0, a1, a2, a3] = a.read_transposed44(4 * k);
      Complex4 wa2 = w * a2;
      Complex4 wa3 = w * a3;
      Complex4 e2 = a0 + wa2;
      Complex4 e3 = a0 - wa2;
      Complex4 wo2 = w2 * (a1 + wa3);
      Complex4 wo3 = w3 * (a1 - wa3);
      a.write_transposed44(4 * k, e2 + wo2, e2 - wo2, e3 + wo3, e3 - wo3);
    }
  };

  auto radix2_high = [&]() {
    for (size_t k = k_base; k < k_base + (1uz << (n_pow - 1 - step)); k++) {
      Complex4 w = twiddles_bitreversed_span.read1(k);
      for (size_t j = (2 * k) << step; j < ((2 * k + 1) << step); j += 4) {
        Complex4 e = a.read4(j);
        Complex4 o = a.read4(j + (1uz << step));
        Complex4 wo = w * o;
        a.write4(j, e + wo);
        a.write4(j + (1uz << step), e - wo);
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
    fft_dif(a, n_pow - 2, k_base);
    fft_dif(a, n_pow - 2, k_base + 1);
    fft_dif(a, n_pow - 2, k_base + 2);
    fft_dif(a, n_pow - 2, k_base + 3);
  } else {
    radix4_low();
  }
}

void ifft_dif(ComplexSpan a, int n_pow, size_t k_base) {
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
  //           Complex e = a.read1(j);
  //           Complex o = a.read1(j + (1uz << step));
  //           Complex eo = e - o;
  //           a.write1(j, e + o);
  //           a.write1(j + (1uz << step), w.conj_mul(eo));
  //         }
  //       }
  //     }

  // Prevent re-reading of twiddles base address after every store, because Clang assumes
  // _mm*_store_* may alias anything
  ComplexSpan twiddles_bitreversed_span(twiddles_bitreversed);
  int step = 0;

  // It's critical for performance to perform multiple steps at once
  auto radix4_high = [&]() {
    for (size_t k = k_base; k < k_base + (1uz << (n_pow - 2 - step)); k++) {
      Complex4 w = twiddles_bitreversed_span.read1(k);
      Complex4 w0 = twiddles_bitreversed_span.read1(2 * k);
      Complex4 w1 = twiddles_bitreversed_span.read1(2 * k + 1);

      size_t i1 = 1uz << step;
      size_t i2 = 2uz << step;
      size_t i3 = 3uz << step;
      for (size_t j = (4 * k) << step; j < ((4 * k + 1) << step); j += 4) {
        Complex4 e0 = a.read4(j);
        Complex4 o0 = a.read4(j + i1);
        Complex4 e1 = a.read4(j + i2);
        Complex4 o1 = a.read4(j + i3);
        Complex4 e2 = e0 + o0;
        Complex4 e3 = w0.conj_mul(e0 - o0);
        Complex4 o2 = e1 + o1;
        Complex4 o3 = w1.conj_mul(e1 - o1);
        a.write4(j, e2 + o2);
        a.write4(j + i1, e3 + o3);
        a.write4(j + i2, w.conj_mul(e2 - o2));
        a.write4(j + i3, w.conj_mul(e3 - o3));
      }
    }
  };

  auto radix4_low = [&]() {
    ensure(step == 0);

    for (size_t k = k_base; k < k_base + (1uz << (n_pow - 2)); k += 4) {
      Complex4 w = twiddles_bitreversed_span.read4(k);
      auto [w0, w1] = twiddles_bitreversed_span.read_transposed24(2 * k);
      auto [e0, o0, e1, o1] = a.read_transposed44(4 * k);
      Complex4 e2 = e0 + o0;
      Complex4 e3 = w0.conj_mul(e0 - o0);
      Complex4 o2 = e1 + o1;
      Complex4 o3 = w1.conj_mul(e1 - o1);
      a.write_transposed44(4 * k, e2 + o2, e3 + o3, w.conj_mul(e2 - o2), w.conj_mul(e3 - o3));
    }
  };

  auto radix2_high = [&]() {
    for (size_t k = k_base; k < k_base + (1uz << (n_pow - 1 - step)); k++) {
      Complex4 w = twiddles_bitreversed_span.read1(k);
      for (size_t j = (2 * k) << step; j < ((2 * k + 1) << step); j += 4) {
        Complex4 e = a.read4(j);
        Complex4 o = a.read4(j + (1uz << step));
        a.write4(j, e + o);
        a.write4(j + (1uz << step), w.conj_mul(e - o));
      }
    }
  };

  // If n_pow >= FFT_RECURSIVE, we want to do recurse and then do one radix-4 step. Otherwise, apply
  // various steps iteratively. We want to only have one invocation of radix4_high so that it is
  // inlined, so the code is a bit more messy than it could be.

  if (n_pow >= FFT_RECURSIVE) {
    ifft_dif(a, n_pow - 2, k_base);
    ifft_dif(a, n_pow - 2, k_base + (1uz << (n_pow - 2)));
    ifft_dif(a, n_pow - 2, k_base + (2uz << (n_pow - 2)));
    ifft_dif(a, n_pow - 2, k_base + (3uz << (n_pow - 2)));
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

// Arithmetic mod 2^64 - 1
struct Mod {
  // For 0, may be either 0 or 2^64 - 1
  uint64_t value;

  Mod operator*(Mod rhs) const {
    __uint128_t prod = (__uint128_t)value * rhs.value;
    unsigned long long result;
    result += __builtin_uaddll_overflow(prod, prod >> 64, &result);
    return {result};
  }
  Mod operator+(Mod rhs) const {
    uint64_t sum;
    sum += __builtin_add_overflow(value, rhs.value, &sum);
    return {sum};
  }
  bool operator==(Mod rhs) const {
    uint64_t diff = value - rhs.value;
    return diff == 0 || diff == -1ULL;
  }
};

template<bool VerificationNoOp>
struct PolynomialEvaluator {
  std::array<Mod, 4> value_vec{0, 0, 0, 0};
  Mod point, pointi, point4;

  PolynomialEvaluator(Mod point) : point(point) {
    pointi = Mod{1};
    point4 = point * point * point * point;
  }

  void supply(__m256i words) {
    value_vec[0] = value_vec[0] + Mod{static_cast<uint64_t>(words[0])} * pointi;
    value_vec[1] = value_vec[1] + Mod{static_cast<uint64_t>(words[1])} * pointi;
    value_vec[2] = value_vec[2] + Mod{static_cast<uint64_t>(words[2])} * pointi;
    value_vec[3] = value_vec[3] + Mod{static_cast<uint64_t>(words[3])} * pointi;
    pointi = pointi * point4;
  }

  Mod get(std::array<int, 4> indexes) const {
    Mod value = value_vec[indexes[3]];
    value = (value * point) + value_vec[indexes[2]];
    value = (value * point) + value_vec[indexes[1]];
    value = (value * point) + value_vec[indexes[0]];
    return value;
  }
};

template<>
struct PolynomialEvaluator<true> {
  PolynomialEvaluator(Mod point) {}
  void supply(__m256i words) {}
  Mod get(std::array<int, 4> indexes) const {
    return {0};
  }
};

template<bool VerificationNoOp>
std::pair<ComplexArray, Mod> mul_fft_transform_input(ConstRef input, int n_pow, int word_bitness, Mod point) {
  size_t n = 1uz << n_pow;

  ComplexArray input_fft(n);

  // Split into words
  size_t k = 0;
  PolynomialEvaluator<VerificationNoOp> ev(point);
  auto write_words = [&](__m256i words) {
    ev.supply(words);
    input_fft.write2(k * 2, Complex2::from_continuous(u64_to_double(words)));
  };
  if (word_bitness == 16) {
    for (; k < input.data.size(); k++) {
      write_words(
        _mm256_blend_epi16(
          _mm256_setzero_si256(),
          _mm256_srlv_epi64(_mm256_set1_epi64x(input.data[k]), _mm256_set_epi64x(48, 16, 32, 0)),
          0x11
        )
      );
    }
  } else if (word_bitness == 12) {
    const uint16_t *data = reinterpret_cast<const uint16_t*>(input.data.data());
    for (; (k + 1) * 3 < input.data.size() * 4; k++) {
      // 48-bit word; it is safe to overread by 1 byte, and we mask it later with shifted 0xfff, so
      // no need to get rid of the top 16 bits
      uint64_t word;
      memcpy(&word, &data[k * 3], 8);
      // Split into 12-bit parts
      write_words(
        _mm256_and_si256(
          _mm256_srlv_epi64(_mm256_set1_epi64x(word), _mm256_set_epi64x(36, 12, 24, 0)),
          _mm256_set1_epi64x(0xfff)
        )
      );
    }

    // Last part:
    // 48-bit word
    uint64_t word = k * 3 + 2 < input.data.size() * 4 ? data[k * 3 + 2] : 0;
    word = (word << 16) | (k * 3 + 1 < input.data.size() * 4 ? data[k * 3 + 1] : 0);
    word = (word << 16) | data[k * 3];
    // Split into 12-bit parts
    write_words(
      _mm256_and_si256(
        _mm256_srlv_epi64(_mm256_set1_epi64x(word), _mm256_set_epi64x(36, 12, 24, 0)),
        _mm256_set1_epi64x(0xfff)
      )
    );
    k++;
  } else {
    ensure(false);
  }

  input_fft.zero_many(k * 2, n - k * 2);

  fft_dif(input_fft, n_pow, 0);

  return {std::move(input_fft), ev.get({0, 2, 1, 3})};
}

ComplexArray mul_fft_middle_end(ComplexArray lhs_fft_dif, ComplexArray rhs_fft_dif, int n_pow) {
  size_t n = 1uz << n_pow;

  ComplexArray prod_fft_dif(n);

  auto handle_iteration_single = [&](size_t k, size_t k_complement) {
    Complex lhs_k = lhs_fft_dif.read1(k);
    Complex rhs_k = rhs_fft_dif.read1(k);
    Complex twiddles_bitreversed_k = twiddles_bitreversed.read1(k / 2);
    prod_fft_dif.write1(
      k,
      lhs_k * rhs_k
      - (
        (k % 2 == 1 ? -twiddles_bitreversed_k : twiddles_bitreversed_k) + 1
      ) * (
        lhs_k.minus_conj(lhs_fft_dif.read1(k_complement))
        * rhs_k.minus_conj(rhs_fft_dif.read1(k_complement))
      ) * 0.25
    );
  };

  auto handle_iteration_vectorized = [&](size_t k, size_t k_complement) {
    Complex4 lhs_k = lhs_fft_dif.read4(k);
    Complex4 rhs_k = rhs_fft_dif.read4(k);
    Complex4 twiddles_bitreversed_k = twiddles_bitreversed.read2(k / 2).duplicate_continuous().alternate_signs();
    Complex4 c = (
      lhs_k.minus_conj(lhs_fft_dif.read4(k_complement).reverse())
      * rhs_k.minus_conj(rhs_fft_dif.read4(k_complement).reverse())
    );
    prod_fft_dif.write4(k, lhs_k.fmsub(rhs_k, twiddles_bitreversed_k.fmadd(c, c) * 0.25));
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

template<bool VerificationNoOp>
Mod mul_fft_transform_output(Ref result, ComplexSpan prod_fft_dif, int n_pow, int word_bitness, Mod point) {
  size_t n = 1uz << n_pow;
  __m256d magic = _mm256_set1_pd(0x1p52 * n);

  ifft_dif(prod_fft_dif, n_pow, 0);

  uint64_t carry = 0;

  PolynomialEvaluator<VerificationNoOp> ev(point);
  auto get_word = [&](size_t k) {
    __m256i word = _mm256_castpd_si256(
      _mm256_xor_pd(
        _mm256_add_pd(
          _mm256_andnot_pd(_mm256_set1_pd(-0.), prod_fft_dif.read2(2 * k).to_interleaved()),
          magic
        ),
        magic
      )
    );
    ev.supply(word);
    return word;
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

  return ev.get({0, 1, 2, 3});
}

std::ostream& operator<<(std::ostream& out, ConstRef rhs);

template<bool VerificationNoOp>
bool mul_fft(Ref result, ConstRef lhs, ConstRef rhs, int n_pow, int word_bitness) {
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
  // Treating the polynomials as mod 2^64 - 1, evaluate them at a point. This ensures that:
  // a) single-coefficient errors are reliably detected,
  // b) arbitrary errors are detected with probability at least 1 - n / (largest prime factor of
  //    2^64 - 1) = 1 - n / 6700417 ~ 0.96 for the upper limit on n for 16 bits,
  // c) random errors are detected with probability 1 - n / (2^64 - 1) = 1 - 1.5e-14.
  // This is enough to counter all known counter-examples and provide sufficient guarantees for
  // unknown counter-examples.
  Mod point;
  if constexpr (!VerificationNoOp) {
    static thread_local std::mt19937_64 point_generator(std::random_device{}());
    point = {point_generator()};
  }
  auto [lhs_fft_dif, lhs_value] = mul_fft_transform_input<VerificationNoOp>(lhs, n_pow, word_bitness, point);
  auto [rhs_fft_dif, rhs_value] = mul_fft_transform_input<VerificationNoOp>(rhs, n_pow, word_bitness, point);
  ComplexArray prod_fft_dif = mul_fft_middle_end(std::move(lhs_fft_dif), std::move(rhs_fft_dif), n_pow);
  Mod prod_value = mul_fft_transform_output<VerificationNoOp>(result.slice(0, lhs.data.size() + rhs.data.size()), prod_fft_dif, n_pow, word_bitness, point);
  return prod_value == lhs_value * rhs_value;
}

BigInt::BigInt(const uint64_t* data, size_t size) : data(data, size) {}

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

#ifdef __ADX__
#define IFADX(yes, no) yes
#else
#define IFADX(yes, no) no
#endif

#define MUL_QUADRATIC_BODY \
  size_t i = 0; \
  uint64_t low, high; \
  \
  if (rhs.data.size() == 1) { \
    uint64_t b = rhs.data[0]; \
    uint64_t tmp = 0; \
    for (; i + 4 <= lhs.data.size(); i += 4) { \
      asm volatile( \
        MOV_OR_ADD("", IFADX("xor %[high], %[high];", "")) /* set CF = OF = 0 */ \
        "mulx %[a0], %[low], %[high];" \
        MOV_OR_ADD( \
          "add %[tmp], %[low]; mov %[low], %[result0];", \
          IFADX( \
            "adcx %[tmp], %[low]; adox %[result0], %[low]; mov %[low], %[result0];", \
            "add %[tmp], %[low]; adc $0, %[high]; add %[low], %[result0];" \
          ) \
        ) \
        "mulx %[a1], %[low], %[tmp];" \
        MOV_OR_ADD( \
          "adc %[high], %[low]; mov %[low], %[result1];", \
          IFADX( \
            "adcx %[high], %[low]; adox %[result1], %[low]; mov %[low], %[result1];", \
            "adc %[high], %[low]; adc $0, %[tmp]; add %[low], %[result1];" \
          ) \
        ) \
        "mulx %[a2], %[low], %[high];" \
        MOV_OR_ADD( \
          "add %[tmp], %[low]; mov %[low], %[result2];", \
          IFADX( \
            "adcx %[tmp], %[low]; adox %[result2], %[low]; mov %[low], %[result2];", \
            "adc %[tmp], %[low]; adc $0, %[high]; add %[low], %[result2];" \
          ) \
        ) \
        "mulx %[a3], %[low], %[tmp];" \
        MOV_OR_ADD( \
          "adc %[high], %[low]; mov %[low], %[result3]; adc $0, %[tmp];", \
          IFADX( \
            "adcx %[high], %[low]; adox %[result3], %[low]; mov %[low], %[result3]; mov $0, %k[low]; seto %b[low]; adcx %[low], %[tmp];", \
            "adc %[high], %[low]; adc $0, %[tmp]; add %[low], %[result3]; adc $0, %[tmp];" \
          ) \
        ) \
        : [low] "=&r"(low), [high] "=&r"(high), [tmp] "+r"(tmp), [result0] "+m"(result.data[i]), [result1] "=&m"(result.data[i + 1]), [result2] "=&m"(result.data[i + 2]), [result3] "=&m"(result.data[i + 3]) \
        : [a0] "m"(lhs.data[i]), [a1] "m"(lhs.data[i + 1]), [a2] "m"(lhs.data[i + 2]), [a3] "m"(lhs.data[i + 3]), "d"(b) \
        : "memory", "flags" \
      ); \
    } \
    for (; i < lhs.data.size(); i++) { \
      asm volatile( \
        "mulx %[a], %[low], %[high];" \
        "add %[tmp], %[low];" \
        "adc $0, %[high];" \
        MOV_OR_ADD( \
          "mov %[low], %[result];", \
          "add %[low], %[result];" \
          "adc $0, %[high];" \
        ) \
        "mov %[high], %[tmp];" \
        : [low] "=&r"(low), [high] "=&r"(high), [tmp] "+r"(tmp), [result] "+m"(result.data[i]) \
        : [a] "m"(lhs.data[i]), "d"(b) \
        : "memory", "flags" \
      ); \
    } \
    result.data[i] = tmp; \
  } else if (rhs.data.size() == 2) { \
    uint64_t b0 = rhs.data[0]; \
    uint64_t b1 = rhs.data[1]; \
    uint64_t tmp0 = 0; \
    uint64_t tmp1 = 0; \
    for (; i < lhs.data.size(); i++) { \
      uint64_t a = lhs.data[i]; \
      asm volatile( \
        IFADX("xor %[high], %[high];", "") /* set CF = OF = 0 */ \
        "mulx %[b0], %[low], %[high];" \
        IFADX( \
          "adcx %[tmp0], %[low];" MOV_OR_ADD("", "adox %[result], %[low];") "mov %[low], %[result];", \
          "add %[tmp0], %[low];" MOV_OR_ADD("mov %[low], %[result];", "adc $0, %[high]; add %[low], %[result];") \
        ) \
        "mulx %[b1], %[tmp0], %[low];" \
        IFADX( \
          "adcx %[high], %[tmp0]; adox %[tmp1], %[tmp0]; mov $0, %k[high]; seto %b[high]; adcx %[high], %[low];", \
          "adc %[high], %[tmp0]; adc $0, %[low]; add %[tmp1], %[tmp0]; adc $0, %[low];" \
        ) \
        "mov %[low], %[tmp1];" \
        : [low] "=&r"(low), [high] "=&r"(high), [tmp0] "+r"(tmp0), [tmp1] "+r"(tmp1), [result] "+m"(result.data[i]) \
        : [b0] "r"(b0), [b1] "r"(b1), "d"(a) \
        : "memory", "flags" \
      ); \
    } \
    result.data[i] = tmp0; \
    result.data[i + 1] = tmp1; \
  } else if (rhs.data.size() == 3) { \
    uint64_t b0 = rhs.data[0]; \
    uint64_t b1 = rhs.data[1]; \
    uint64_t b2 = rhs.data[2]; \
    uint64_t tmp0 = 0; \
    uint64_t tmp1 = 0; \
    uint64_t tmp2 = 0; \
    for (; i < lhs.data.size(); i++) { \
      uint64_t a = lhs.data[i]; \
      asm volatile( \
        IFADX("xor %[high], %[high];", "") /* set CF = OF = 0 */ \
        "mulx %[b0], %[low], %[high];" \
        IFADX( \
          "adcx %[tmp0], %[low];" MOV_OR_ADD("", "adox %[result], %[low];") "mov %[low], %[result];", \
          "add %[tmp0], %[low];" MOV_OR_ADD("mov %[low], %[result];", "adc $0, %[high]; add %[low], %[result];") \
        ) \
        "mulx %[b1], %[tmp0], %[low];" \
        IFADX( \
          "adcx %[high], %[tmp0]; adox %[tmp1], %[tmp0];", \
          "adc %[high], %[tmp0]; adc $0, %[low]; add %[tmp1], %[tmp0];" \
        ) \
        "mulx %[b2], %[tmp1], %[high];" \
        IFADX( \
          "adcx %[low], %[tmp1]; adox %[tmp2], %[tmp1]; mov $0, %k[low]; seto %b[low]; adcx %[low], %[high];", \
          "adc %[low], %[tmp1]; adc $0, %[high]; add %[tmp2], %[tmp1]; adc $0, %[high];" \
        ) \
        "mov %[high], %[tmp2];" \
        : [low] "=&r"(low), [high] "=&r"(high), [tmp0] "+r"(tmp0), [tmp1] "+r"(tmp1), [tmp2] "+r"(tmp2), [result] "+m"(result.data[i]) \
        : [b0] "r"(b0), [b1] "r"(b1), [b2] "r"(b2), "d"(a) \
        : "memory", "flags" \
      ); \
    } \
    result.data[i] = tmp0; \
    result.data[i + 1] = tmp1; \
    result.data[i + 2] = tmp2; \
  } else if (rhs.data.size() == 4) { \
    uint64_t b0 = rhs.data[0]; \
    uint64_t b1 = rhs.data[1]; \
    uint64_t b2 = rhs.data[2]; \
    uint64_t b3 = rhs.data[3]; \
    uint64_t tmp0 = 0; \
    uint64_t tmp1 = 0; \
    uint64_t tmp2 = 0; \
    uint64_t tmp3 = 0; \
    for (; i < lhs.data.size(); i++) { \
      uint64_t a = lhs.data[i]; \
      asm volatile( \
        IFADX("xor %[high], %[high];", "") /* set CF = OF = 0 */ \
        "mulx %[b0], %[low], %[high];" \
        IFADX( \
          "adcx %[tmp0], %[low];" MOV_OR_ADD("", "adox %[result], %[low];") "mov %[low], %[result];", \
          "add %[tmp0], %[low];" MOV_OR_ADD("mov %[low], %[result];", "adc $0, %[high]; add %[low], %[result];") \
        ) \
        "mulx %[b1], %[tmp0], %[low];" \
        IFADX( \
          "adcx %[high], %[tmp0]; adox %[tmp1], %[tmp0];", \
          "adc %[high], %[tmp0]; adc $0, %[low]; add %[tmp1], %[tmp0];" \
        ) \
        "mulx %[b2], %[tmp1], %[high];" \
        IFADX( \
          "adcx %[low], %[tmp1]; adox %[tmp2], %[tmp1];", \
          "adc %[low], %[tmp1]; adc $0, %[high]; add %[tmp2], %[tmp1];" \
        ) \
        "mulx %[b3], %[tmp2], %[low];" \
        IFADX( \
          "adcx %[high], %[tmp2]; adox %[tmp3], %[tmp2]; mov $0, %k[high]; seto %b[high]; adcx %[high], %[low];", \
          "adc %[high], %[tmp2]; adc $0, %[low]; add %[tmp3], %[tmp2]; adc $0, %[low];" \
        ) \
        "mov %[low], %[tmp3];" \
        : [low] "=&r"(low), [high] "=&r"(high), [tmp0] "+r"(tmp0), [tmp1] "+r"(tmp1), [tmp2] "+r"(tmp2), [tmp3] "+r"(tmp3), [result] "+m"(result.data[i]) \
        : [b0] "r"(b0), [b1] "r"(b1), [b2] "r"(b2), [b3] "r"(b3), "d"(a) \
        : "memory", "flags" \
      ); \
    } \
    result.data[i] = tmp0; \
    result.data[i + 1] = tmp1; \
    result.data[i + 2] = tmp2; \
    result.data[i + 3] = tmp3; \
  } else if (rhs.data.size() == 5) { \
    /* We don't have enough general-purpose registers in this case, so b4 and tmp4 go to XMM \
       registers */ \
    uint64_t b0 = rhs.data[0]; \
    uint64_t b1 = rhs.data[1]; \
    uint64_t b2 = rhs.data[2]; \
    uint64_t b3 = rhs.data[3]; \
    __m128i b4 = _mm_cvtsi64_si128(rhs.data[4]); \
    uint64_t tmp0 = 0; \
    uint64_t tmp1 = 0; \
    uint64_t tmp2 = 0; \
    uint64_t tmp3 = 0; \
    __m128i tmp4 = _mm_setzero_si128(); \
    for (; i < lhs.data.size(); i++) { \
      uint64_t a = lhs.data[i]; \
      asm volatile( \
        IFADX("xor %[high], %[high];", "") /* set CF = OF = 0 */ \
        "mulx %[b0], %[low], %[high];" \
        IFADX( \
          "adcx %[tmp0], %[low];" MOV_OR_ADD("", "adox %[result], %[low];") "mov %[low], %[result];", \
          "add %[tmp0], %[low];" MOV_OR_ADD("mov %[low], %[result];", "adc $0, %[high]; add %[low], %[result];") \
        ) \
        "mulx %[b1], %[tmp0], %[low];" \
        IFADX( \
          "adcx %[high], %[tmp0]; adox %[tmp1], %[tmp0];", \
          "adc %[high], %[tmp0]; adc $0, %[low]; add %[tmp1], %[tmp0];" \
        ) \
        "mulx %[b2], %[tmp1], %[high];" \
        IFADX( \
          "adcx %[low], %[tmp1]; adox %[tmp2], %[tmp1];", \
          "adc %[low], %[tmp1]; adc $0, %[high]; add %[tmp2], %[tmp1];" \
        ) \
        "mulx %[b3], %[tmp2], %[low];" \
        IFADX( \
          "adcx %[high], %[tmp2]; adox %[tmp3], %[tmp2];", \
          "adc %[high], %[tmp2]; adc $0, %[low]; add %[tmp3], %[tmp2];" \
        ) \
        "movq %[b4], %[high];" \
        "mulx %[high], %[tmp3], %[high];" \
        IFADX( \
          "adcx %[low], %[tmp3]; movq %[tmp4], %[low]; adox %[low], %[tmp3]; mov $0, %k[low]; seto %b[low]; adcx %[low], %[high];", \
          "adc %[low], %[tmp3]; adc $0, %[high]; movq %[tmp4], %[low]; add %[low], %[tmp3]; adc $0, %[high];" \
        ) \
        "movq %[high], %[tmp4];" \
        : [low] "=&r"(low), [high] "=&r"(high), [tmp0] "+r"(tmp0), [tmp1] "+r"(tmp1), [tmp2] "+r"(tmp2), [tmp3] "+r"(tmp3), [tmp4] "+x"(tmp4), [result] "+m"(result.data[i]) \
        : [b0] "r"(b0), [b1] "r"(b1), [b2] "r"(b2), [b3] "r"(b3), [b4] "x"(b4), "d"(a) \
        : "memory", "flags" \
      ); \
    } \
    result.data[i] = tmp0; \
    result.data[i + 1] = tmp1; \
    result.data[i + 2] = tmp2; \
    result.data[i + 3] = tmp3; \
    result.data[i + 4] = _mm_cvtsi128_si64(tmp4); \
  } else if (rhs.data.size() == 6) { \
    /* We don't have enough general-purpose registers in this case, so b3, b4, b5 and tmp5 go to XMM \
       registers */ \
    uint64_t b0 = rhs.data[0]; \
    uint64_t b1 = rhs.data[1]; \
    uint64_t b2 = rhs.data[2]; \
    __m128i b3 = _mm_cvtsi64_si128(rhs.data[3]); \
    __m128i b4 = _mm_cvtsi64_si128(rhs.data[4]); \
    __m128i b5 = _mm_cvtsi64_si128(rhs.data[5]); \
    uint64_t tmp0 = 0; \
    uint64_t tmp1 = 0; \
    uint64_t tmp2 = 0; \
    uint64_t tmp3 = 0; \
    uint64_t tmp4 = 0; \
    __m128i tmp5 = _mm_setzero_si128(); \
    for (; i < lhs.data.size(); i++) { \
      uint64_t a = lhs.data[i]; \
      asm volatile( \
        IFADX("xor %[high], %[high];", "") /* set CF = OF = 0 */ \
        "mulx %[b0], %[low], %[high];" \
        IFADX( \
          "adcx %[tmp0], %[low];" MOV_OR_ADD("", "adox %[result], %[low];") "mov %[low], %[result];", \
          "add %[tmp0], %[low];" MOV_OR_ADD("mov %[low], %[result];", "adc $0, %[high]; add %[low], %[result];") \
        ) \
        "mulx %[b1], %[tmp0], %[low];" \
        IFADX( \
          "adcx %[high], %[tmp0]; adox %[tmp1], %[tmp0];", \
          "adc %[high], %[tmp0]; adc $0, %[low]; add %[tmp1], %[tmp0];" \
        ) \
        "mulx %[b2], %[tmp1], %[high];" \
        IFADX( \
          "adcx %[low], %[tmp1]; adox %[tmp2], %[tmp1];", \
          "adc %[low], %[tmp1]; adc $0, %[high]; add %[tmp2], %[tmp1];" \
        ) \
        "movq %[b3], %[low];" \
        "mulx %[low], %[tmp2], %[low];" \
        IFADX( \
          "adcx %[high], %[tmp2]; adox %[tmp3], %[tmp2];", \
          "adc %[high], %[tmp2]; adc $0, %[low]; add %[tmp3], %[tmp2];" \
        ) \
        "movq %[b4], %[high];" \
        "mulx %[high], %[tmp3], %[high];" \
        IFADX( \
          "adcx %[low], %[tmp3]; adox %[tmp4], %[tmp3];", \
          "adc %[low], %[tmp3]; adc $0, %[high]; add %[tmp4], %[tmp3];" \
        ) \
        "movq %[b5], %[low];" \
        "mulx %[low], %[tmp4], %[low];" \
        IFADX( \
          "adcx %[high], %[tmp4]; movq %[tmp5], %[high]; adox %[high], %[tmp4]; mov $0, %k[high]; seto %b[high]; adcx %[high], %[low];", \
          "adc %[high], %[tmp4]; adc $0, %[low]; movq %[tmp5], %[high]; add %[high], %[tmp4]; adc $0, %[low];" \
        ) \
        "movq %[low], %[tmp5];" \
        : [low] "=&r"(low), [high] "=&r"(high), [tmp0] "+r"(tmp0), [tmp1] "+r"(tmp1), [tmp2] "+r"(tmp2), [tmp3] "+r"(tmp3), [tmp4] "+r"(tmp4), [tmp5] "+x"(tmp5), [result] "+m"(result.data[i]) \
        : [b0] "r"(b0), [b1] "r"(b1), [b2] "r"(b2), [b3] "x"(b3), [b4] "x"(b4), [b5] "x"(b5), "d"(a) \
        : "memory", "flags" \
      ); \
    } \
    result.data[i] = tmp0; \
    result.data[i + 1] = tmp1; \
    result.data[i + 2] = tmp2; \
    result.data[i + 3] = tmp3; \
    result.data[i + 4] = tmp4; \
    result.data[i + 5] = _mm_cvtsi128_si64(tmp5); \
  } else if (rhs.data.size() == 7) { \
    /* We don't have enough general-purpose registers in this case, so b2, b3, b4, b5, b6 and tmp6 \
       go to XMM registers */ \
    uint64_t b0 = rhs.data[0]; \
    uint64_t b1 = rhs.data[1]; \
    __m128i b2 = _mm_cvtsi64_si128(rhs.data[2]); \
    __m128i b3 = _mm_cvtsi64_si128(rhs.data[3]); \
    __m128i b4 = _mm_cvtsi64_si128(rhs.data[4]); \
    __m128i b5 = _mm_cvtsi64_si128(rhs.data[5]); \
    __m128i b6 = _mm_cvtsi64_si128(rhs.data[6]); \
    uint64_t tmp0 = 0; \
    uint64_t tmp1 = 0; \
    uint64_t tmp2 = 0; \
    uint64_t tmp3 = 0; \
    uint64_t tmp4 = 0; \
    uint64_t tmp5 = 0; \
    __m128i tmp6 = _mm_setzero_si128(); \
    for (; i < lhs.data.size(); i++) { \
      uint64_t a = lhs.data[i]; \
      asm volatile( \
        IFADX("xor %[high], %[high];", "") /* set CF = OF = 0 */ \
        "mulx %[b0], %[low], %[high];" \
        IFADX( \
          "adcx %[tmp0], %[low];" MOV_OR_ADD("", "adox %[result], %[low];") "mov %[low], %[result];", \
          "add %[tmp0], %[low];" MOV_OR_ADD("mov %[low], %[result];", "adc $0, %[high]; add %[low], %[result];") \
        ) \
        "mulx %[b1], %[tmp0], %[low];" \
        IFADX( \
          "adcx %[high], %[tmp0]; adox %[tmp1], %[tmp0];", \
          "adc %[high], %[tmp0]; adc $0, %[low]; add %[tmp1], %[tmp0];" \
        ) \
        "movq %[b2], %[high];" \
        "mulx %[high], %[tmp1], %[high];" \
        IFADX( \
          "adcx %[low], %[tmp1]; adox %[tmp2], %[tmp1];", \
          "adc %[low], %[tmp1]; adc $0, %[high]; add %[tmp2], %[tmp1];" \
        ) \
        "movq %[b3], %[low];" \
        "mulx %[low], %[tmp2], %[low];" \
        IFADX( \
          "adcx %[high], %[tmp2]; adox %[tmp3], %[tmp2];", \
          "adc %[high], %[tmp2]; adc $0, %[low]; add %[tmp3], %[tmp2];" \
        ) \
        "movq %[b4], %[high];" \
        "mulx %[high], %[tmp3], %[high];" \
        IFADX( \
          "adcx %[low], %[tmp3]; adox %[tmp4], %[tmp3];", \
          "adc %[low], %[tmp3]; adc $0, %[high]; add %[tmp4], %[tmp3];" \
        ) \
        "movq %[b5], %[low];" \
        "mulx %[low], %[tmp4], %[low];" \
        IFADX( \
          "adcx %[high], %[tmp4]; adox %[tmp5], %[tmp4];", \
          "adc %[high], %[tmp4]; adc $0, %[low]; add %[tmp5], %[tmp4];" \
        ) \
        "movq %[b6], %[high];" \
        "mulx %[high], %[tmp5], %[high];" \
        IFADX( \
          "adcx %[low], %[tmp5]; movq %[tmp6], %[low]; adox %[low], %[tmp5]; mov $0, %k[low]; seto %b[low]; adcx %[low], %[high];", \
          "adc %[low], %[tmp5]; adc $0, %[high]; movq %[tmp6], %[low]; add %[low], %[tmp5]; adc $0, %[high];" \
        ) \
        "movq %[high], %[tmp6];" \
        : [low] "=&r"(low), [high] "=&r"(high), [tmp0] "+r"(tmp0), [tmp1] "+r"(tmp1), [tmp2] "+r"(tmp2), [tmp3] "+r"(tmp3), [tmp4] "+r"(tmp4), [tmp5] "+r"(tmp5), [tmp6] "+x"(tmp6), [result] "+m"(result.data[i]) \
        : [b0] "r"(b0), [b1] "r"(b1), [b2] "x"(b2), [b3] "x"(b3), [b4] "x"(b4), [b5] "x"(b5), [b6] "x"(b6), "d"(a) \
        : "memory", "flags" \
      ); \
    } \
    result.data[i] = tmp0; \
    result.data[i + 1] = tmp1; \
    result.data[i + 2] = tmp2; \
    result.data[i + 3] = tmp3; \
    result.data[i + 4] = tmp4; \
    result.data[i + 5] = tmp5; \
    result.data[i + 6] = _mm_cvtsi128_si64(tmp6); \
  } else if (rhs.data.size() == 8) { \
    /* We don't have enough general-purpose registers in this case, so b0, b1, b2, b3, b4, b5, b6,
       b7 and tmp7 go to XMM registers */ \
    __m128i b0 = _mm_cvtsi64_si128(rhs.data[0]); \
    __m128i b1 = _mm_cvtsi64_si128(rhs.data[1]); \
    __m128i b2 = _mm_cvtsi64_si128(rhs.data[2]); \
    __m128i b3 = _mm_cvtsi64_si128(rhs.data[3]); \
    __m128i b4 = _mm_cvtsi64_si128(rhs.data[4]); \
    __m128i b5 = _mm_cvtsi64_si128(rhs.data[5]); \
    __m128i b6 = _mm_cvtsi64_si128(rhs.data[6]); \
    __m128i b7 = _mm_cvtsi64_si128(rhs.data[7]); \
    uint64_t tmp0 = 0; \
    uint64_t tmp1 = 0; \
    uint64_t tmp2 = 0; \
    uint64_t tmp3 = 0; \
    uint64_t tmp4 = 0; \
    uint64_t tmp5 = 0; \
    uint64_t tmp6 = 0; \
    __m128i tmp7 = _mm_setzero_si128(); \
    for (; i < lhs.data.size(); i++) { \
      uint64_t a = lhs.data[i]; \
      asm volatile( \
        IFADX("xor %[high], %[high];", "") /* set CF = OF = 0 */ \
        "movq %[b0], %[high];" \
        "mulx %[high], %[low], %[high];" \
        IFADX( \
          "adcx %[tmp0], %[low];" MOV_OR_ADD("", "adox %[result], %[low];") "mov %[low], %[result];", \
          "add %[tmp0], %[low];" MOV_OR_ADD("mov %[low], %[result];", "adc $0, %[high]; add %[low], %[result];") \
        ) \
        "movq %[b1], %[low];" \
        "mulx %[low], %[tmp0], %[low];" \
        IFADX( \
          "adcx %[high], %[tmp0]; adox %[tmp1], %[tmp0];", \
          "adc %[high], %[tmp0]; adc $0, %[low]; add %[tmp1], %[tmp0];" \
        ) \
        "movq %[b2], %[high];" \
        "mulx %[high], %[tmp1], %[high];" \
        IFADX( \
          "adcx %[low], %[tmp1]; adox %[tmp2], %[tmp1];", \
          "adc %[low], %[tmp1]; adc $0, %[high]; add %[tmp2], %[tmp1];" \
        ) \
        "movq %[b3], %[low];" \
        "mulx %[low], %[tmp2], %[low];" \
        IFADX( \
          "adcx %[high], %[tmp2]; adox %[tmp3], %[tmp2];", \
          "adc %[high], %[tmp2]; adc $0, %[low]; add %[tmp3], %[tmp2];" \
        ) \
        "movq %[b4], %[high];" \
        "mulx %[high], %[tmp3], %[high];" \
        IFADX( \
          "adcx %[low], %[tmp3]; adox %[tmp4], %[tmp3];", \
          "adc %[low], %[tmp3]; adc $0, %[high]; add %[tmp4], %[tmp3];" \
        ) \
        "movq %[b5], %[low];" \
        "mulx %[low], %[tmp4], %[low];" \
        IFADX( \
          "adcx %[high], %[tmp4]; adox %[tmp5], %[tmp4];", \
          "adc %[high], %[tmp4]; adc $0, %[low]; add %[tmp5], %[tmp4];" \
        ) \
        "movq %[b6], %[high];" \
        "mulx %[high], %[tmp5], %[high];" \
        IFADX( \
          "adcx %[low], %[tmp5]; adox %[tmp6], %[tmp5];", \
          "adc %[low], %[tmp5]; adc $0, %[high]; add %[tmp6], %[tmp5];" \
        ) \
        "movq %[b7], %[low];" \
        "mulx %[low], %[tmp6], %[low];" \
        IFADX( \
          "adcx %[high], %[tmp6]; movq %[tmp7], %[high]; adox %[high], %[tmp6]; mov $0, %k[high]; seto %b[high]; adcx %[high], %[low];", \
          "adc %[high], %[tmp6]; adc $0, %[low]; movq %[tmp7], %[high]; add %[high], %[tmp6]; adc $0, %[low];" \
        ) \
        "movq %[low], %[tmp7];" \
        : [low] "=&r"(low), [high] "=&r"(high), [tmp0] "+r"(tmp0), [tmp1] "+r"(tmp1), [tmp2] "+r"(tmp2), [tmp3] "+r"(tmp3), [tmp4] "+r"(tmp4), [tmp5] "+r"(tmp5), [tmp6] "+r"(tmp6), [tmp7] "+x"(tmp7), [result] "+m"(result.data[i]) \
        : [b0] "x"(b0), [b1] "x"(b1), [b2] "x"(b2), [b3] "x"(b3), [b4] "x"(b4), [b5] "x"(b5), [b6] "x"(b6), [b7] "x"(b7), "d"(a) \
        : "memory", "flags" \
      ); \
    } \
    result.data[i] = tmp0; \
    result.data[i + 1] = tmp1; \
    result.data[i + 2] = tmp2; \
    result.data[i + 3] = tmp3; \
    result.data[i + 4] = tmp4; \
    result.data[i + 5] = tmp5; \
    result.data[i + 6] = tmp6; \
    result.data[i + 7] = _mm_cvtsi128_si64(tmp7); \
  } else { \
    MOV_OR_ADD(mul_quadratic, fma_quadratic)(result.slice(0), lhs, rhs.slice(0, 8)); \
    size_t j = 8; \
    for (; j + 8 < rhs.data.size(); j += 8) { \
      fma_quadratic(result.slice(j), lhs, rhs.slice(j, 8)); \
    } \
    fma_quadratic(result.slice(j), lhs, rhs.slice(j)); \
  }

void fma_quadratic(Ref result, ConstRef lhs, ConstRef rhs) {
#define MOV_OR_ADD(mov, add) add
  MUL_QUADRATIC_BODY
#undef MOV_OR_ADD
}
void mul_quadratic(Ref result, ConstRef lhs, ConstRef rhs) {
#define MOV_OR_ADD(mov, add) mov
  MUL_QUADRATIC_BODY
#undef MOV_OR_ADD
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
  } else if (std::min(lhs.data.size(), rhs.data.size()) < 40) {
    mul_quadratic(result, lhs, rhs);
  } else {
    if (lhs.data.size() + rhs.data.size() >= 320) {
      int n_pow = get_fft_n_pow_16bit(lhs, rhs);
      if (n_pow <= FFT_MAX_16BIT_RELIABLE) {
        // Small enough to be precise if input is split into 16-bit words
        mul_fft<true>(result, lhs, rhs, n_pow, 16);
        return;
      }
      if (n_pow <= FFT_MAX_16BIT) {
        // Small enough to be precise almost always if input is split into 16-bit words
        if (mul_fft<false>(result, lhs, rhs, n_pow, 16)) {
          return;
        }
        // Possibly malicious input, check failed
      }
      // Have to use 12 bits
      n_pow = get_fft_n_pow_12bit(lhs, rhs);
      ensure(n_pow <= FFT_MAX_12BIT);  // XXX: how do we handle larger numbers?
      mul_fft<true>(result, lhs, rhs, n_pow, 12);
      // No known counter-examples
      return;
    }
    if (lhs.data.size() * 2 < rhs.data.size()) {
      mul_disproportional(result, lhs, rhs);
    } else if (rhs.data.size() * 2 < lhs.data.size()) {
      mul_disproportional(result, rhs, lhs);
    } else {
      mul_karatsuba(result, lhs, rhs);
    }
  }

  // ensure((lhs % 179) * (rhs % 179) % 179 == result.slice(0, lhs.data.size() + rhs.data.size()) % 179);
}

BigInt operator*(ConstRef lhs, ConstRef rhs) {
  if (lhs.data.empty() || rhs.data.empty()) {
    return {};
  }
  BigInt result;
  result.data.overwrite_zerofill_leaking(lhs.data.size() + rhs.data.size());
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

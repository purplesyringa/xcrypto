#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <stdint.h>
#include <string_view>
#include <vector>

// namespace libdivide {

// class divisor {
//   uint64_t magic;
//   uint8_t more;

// public:
//   divisor() {}

//   divisor(uint64_t d) {
//     uint8_t floor_log_2_d = 63 - __builtin_clzll(d);
//     // Power of 2
//     if ((d & (d - 1)) == 0) {
//       magic = 0;
//       more = floor_log_2_d;
//     } else {
//       uint64_t proposed_m, rem;
//       asm("divq %[v]"
//           : "=a"(proposed_m), "=d"(rem)
//           : [v] "r"(d), "a"(0), "d"((uint64_t)1 << floor_log_2_d)
//           : "flags");
//       uint64_t e = d - rem;
//       if (e < ((uint64_t)1 << floor_log_2_d)) {
//         more = floor_log_2_d;
//       } else {
//         proposed_m += proposed_m;
//         uint64_t twice_rem = rem + rem;
//         if (twice_rem >= d || twice_rem < rem)
//           proposed_m++;
//         more = floor_log_2_d | 0x80;
//       }
//       magic = 1 + proposed_m;
//     }
//   }

//   friend uint64_t operator/(uint64_t numer, const divisor &denom) {
//     uint8_t more = denom.more;
//     if (denom.magic == 0) {
//       return numer >> more;
//     }
//     uint64_t q = ((__uint128_t)denom.magic * numer) >> 64;
//     if (more & 0x80) {
//       uint64_t t = ((numer - q) >> 1) + q;
//       return t >> (more & 0x7f);
//     } else {
//       return q >> more;
//     }
//   }
// };

// } // namespace libdivide

namespace bigint {

struct with_base {
  uint64_t base;
};

class SmallVec {
  static constexpr size_t INLINE_STORAGE_SIZE = 8;

  uint64_t *_begin;
  size_t _size;
  size_t _capacity;
  uint64_t _inline_storage[INLINE_STORAGE_SIZE];

  void increase_capacity_to(size_t new_capacity) {
    uint64_t *new_begin = new uint64_t[new_capacity];
    std::copy(_begin, _begin + _size, new_begin);
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
      _begin = new uint64_t[list.size()];
      _capacity = list.size();
    }
    std::copy(list.begin(), list.end(), _begin);
    _size = list.size();
  }
  SmallVec(const uint64_t *data, size_t size) {
    if (size <= INLINE_STORAGE_SIZE) {
      _begin = _inline_storage;
      _capacity = INLINE_STORAGE_SIZE;
    } else {
      _begin = new uint64_t[size];
      _capacity = size;
    }
    std::copy(data, data + size, _begin);
    _size = size;
  }

  void init(uint64_t *data, size_t size) {
    _begin = data;
    _size = size;
    _capacity = 0;
  }
  void forget() {
    _begin = _inline_storage;
    _size = 0;
    _capacity = INLINE_STORAGE_SIZE;
  }

  SmallVec(const SmallVec &rhs) {
    if (rhs._size <= INLINE_STORAGE_SIZE) {
      _begin = _inline_storage;
      _capacity = INLINE_STORAGE_SIZE;
    } else {
      _begin = new uint64_t[rhs._size];
      _capacity = rhs._size;
    }
    std::copy(rhs._begin, rhs._begin + rhs._size, _begin);
    _size = rhs._size;
  }
  SmallVec(SmallVec &&rhs) {
    if (rhs._begin == rhs._inline_storage) {
      _begin = _inline_storage;
      std::copy(rhs._begin, rhs._begin + rhs._size, _begin);
    } else {
      _begin = rhs._begin;
    }
    _size = rhs._size;
    _capacity = rhs._capacity;
    rhs._begin = rhs._inline_storage;
    rhs._size = 0;
    rhs._capacity = 0;
  }

  SmallVec &operator=(const SmallVec &rhs) {
    if (rhs._size > _capacity) {
      if (_begin != _inline_storage) {
        delete[] _begin;
      }
      _begin = new uint64_t[rhs._size];
      _capacity = rhs._size;
    }
    std::copy(rhs._begin, rhs._begin + rhs._size, _begin);
    _size = rhs._size;
    return *this;
  }
  SmallVec &operator=(SmallVec &&rhs) {
    if (_begin != _inline_storage) {
      delete[] _begin;
    }
    if (rhs._begin == rhs._inline_storage) {
      _begin = _inline_storage;
      _capacity = INLINE_STORAGE_SIZE;
      _size = rhs._size;
      std::copy(rhs._begin, rhs._begin + rhs._size, _begin);
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
    std::fill(_begin + _size, _begin + new_size, 0);
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

class BigInt {
  SmallVec data;

  template <typename Iterator, typename Map>
  static uint64_t str_to_int_64(Iterator begin, Iterator end, uint64_t base,
                                Map map) {
    uint64_t val = 0;
    for (auto it = end; it != begin;) {
      val *= base;
      val += map(*--it);
    }
    return val;
  }

  template <typename Iterator, typename Map>
  static __uint128_t str_to_int_128(Iterator begin, Iterator end, uint64_t base,
                                    int max_block_len, uint64_t base_product,
                                    Map map) {
    uint64_t low = str_to_int_64(begin, begin + max_block_len, base, map);
    uint64_t high = str_to_int_64(begin + max_block_len, end, base, map);
    return static_cast<__uint128_t>(high) * base_product + low;
  }

  template <typename Iterator, typename Map>
  static void str_to_int(Iterator begin, Iterator end, uint64_t base, Map map,
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
    str_to_int(mid, end, base, map, powers_of_base, max_block_len, base_product,
               high);
    result += high * powers_of_base[low_len_pow];
    str_to_int(begin, mid, base, map, powers_of_base, max_block_len,
               base_product, result);
  }

  template <typename Iterator, typename Map>
  static BigInt str_to_int(Iterator begin, Iterator end, uint64_t base,
                           Map map) {
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
    str_to_int(begin, end, base, map, powers_of_base.data(), max_block_len,
               base_product, result);
    return result;
  }

  static void add_at_no_resize(BigInt &lhs, const BigInt &rhs,
                               size_t lhs_offset) {
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
        : [data_ptr] "r"(lhs.data.data() + lhs_offset), [rhs_data_ptr] "r"(
                                                            rhs.data.data())
        : "flags", "memory");
  }

  static void mul_1x1(BigInt &result, const BigInt &lhs, const BigInt &rhs,
                      size_t offset) {
    __uint128_t product = __uint128_t{lhs.data[0]} * rhs.data[0];
    result.data[offset] = static_cast<uint64_t>(product);
    if (product >= (__uint128_t{1} << 64)) {
      result.data[offset + 1] = static_cast<uint64_t>(product >> 64);
    }
  }

  static void mul_nx1(BigInt &result, const BigInt &lhs, uint64_t rhs,
                      size_t offset) {
    uint64_t carry = 0;
    for (size_t i = 0; i < lhs.data.size(); i++) {
      __uint128_t total = __uint128_t{lhs.data[i]} * rhs + carry;
      result.data[offset + i] = static_cast<uint64_t>(total);
      carry = static_cast<uint64_t>(total >> 64);
    }
    if (carry != 0) {
      result.data[offset + lhs.data.size()] = carry;
    }
  }

  __attribute__((noinline)) static void mul_quadratic(BigInt &result,
                                                      const BigInt &lhs,
                                                      const BigInt &rhs,
                                                      size_t offset) {
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
  } while (0);

      while (left + 8 <= right) {
        LOOP LOOP LOOP LOOP LOOP LOOP LOOP LOOP
      }
      while (left < right) {
        LOOP
      }

      result.data[offset++] = sum_low;
      carry_low = sum_mid;
      carry_high = sum_high;
    }

    if (carry_high > 0) {
      result.data[offset++] = carry_low;
      result.data[offset++] = carry_high;
    } else if (carry_low > 0) {
      result.data[offset++] = carry_low;
    }
  }

  static void mul_karatsuba(BigInt &result, const BigInt &lhs,
                            const BigInt &rhs, size_t offset) {
    size_t b = std::min(lhs.data.size(), rhs.data.size()) / 2;

    BigInt x0, x1, y0, y1;
    x0.data.init(const_cast<uint64_t *>(lhs.data.data()), b);
    x1.data.init(const_cast<uint64_t *>(lhs.data.data()) + b,
                 lhs.data.size() - b);
    y0.data.init(const_cast<uint64_t *>(rhs.data.data()), b);
    y1.data.init(const_cast<uint64_t *>(rhs.data.data()) + b,
                 rhs.data.size() - b);
    while (!x0.data.empty() && x0.data.back() == 0) {
      x0.data.pop_back();
    }
    while (!y0.data.empty() && y0.data.back() == 0) {
      y0.data.pop_back();
    }

    mul_at(result, x0, y0, offset);
    size_t z0_len = 0;
    if (!x0.data.empty() && !y0.data.empty()) {
      z0_len = x0.data.size() + y0.data.size() - 1;
      if (result.data[offset + z0_len] != 0) {
        z0_len++;
      }
    }
    BigInt z0;
    z0.data.init(result.data.data() + offset, z0_len);

    mul_at(result, x1, y1, offset + b * 2);
    size_t z2_len = x1.data.size() + y1.data.size() - 1;
    if (result.data[offset + b * 2 + z2_len] != 0) {
      z2_len++;
    }
    BigInt z2;
    z2.data.init(result.data.data() + offset + b * 2, z2_len);

    BigInt z1 = (x0 + x1) * (y0 + y1) - z0 - z2;
    add_at_no_resize(result, z1, offset + b);

    x0.data.forget();
    x1.data.forget();
    y0.data.forget();
    y1.data.forget();
    z0.data.forget();
    z2.data.forget();
  }

  static void mul_disproportional(BigInt &result, const BigInt &lhs,
                                  const BigInt &rhs, size_t offset) {
    assert(lhs.data.size() < rhs.data.size());

    BigInt rhs_chunk;

    rhs_chunk.data.init(const_cast<uint64_t *>(rhs.data.data()),
                        lhs.data.size());
    while (!rhs_chunk.data.empty() && rhs_chunk.data.back() == 0) {
      rhs_chunk.data.pop_back();
    }
    mul_at(result, lhs, rhs_chunk, offset);

    size_t i = lhs.data.size();
    for (; i + lhs.data.size() < rhs.data.size(); i += lhs.data.size()) {
      rhs_chunk.data.init(const_cast<uint64_t *>(rhs.data.data()) + i,
                          lhs.data.size());
      while (!rhs_chunk.data.empty() && rhs_chunk.data.back() == 0) {
        rhs_chunk.data.pop_back();
      }
      add_at_no_resize(result, lhs * rhs_chunk, offset + i);
    }

    rhs_chunk.data.init(const_cast<uint64_t *>(rhs.data.data()) + i,
                        rhs.data.size() - i);
    add_at_no_resize(result, lhs * rhs_chunk, offset + i);

    rhs_chunk.data.forget();
  }

  static void mul_at(BigInt &result, const BigInt &lhs, const BigInt &rhs,
                     size_t offset) {
    if (lhs.data.empty() || rhs.data.empty()) {
      return;
    }

    if (lhs.data.size() == 1 && rhs.data.size() == 1) {
      mul_1x1(result, lhs, rhs, offset);
    } else if (rhs.data.size() == 1) {
      mul_nx1(result, lhs, rhs.data[0], offset);
    } else if (lhs.data.size() == 1) {
      mul_nx1(result, rhs, lhs.data[0], offset);
    } else if (std::min(lhs.data.size(), rhs.data.size()) >= 40) {
      if (lhs.data.size() * 2 < rhs.data.size()) {
        mul_disproportional(result, lhs, rhs, offset);
      } else if (rhs.data.size() * 2 < lhs.data.size()) {
        mul_disproportional(result, rhs, lhs, offset);
      } else {
        mul_karatsuba(result, lhs, rhs, offset);
      }
    } else {
      mul_quadratic(result, lhs, rhs, offset);
    }
  }

  // j = -i
  static __m256d mul_by_j(__m256d vec) {
    return _mm256_xor_pd(_mm256_permute_pd(vec, 5),
                         _mm256_set_pd(-0., 0., -0., 0.));
  }

  static __m256d load_w(size_t n, size_t cos, size_t sin) {
    __m128d reals = _mm_load_pd(&cosines[n + cos]);
    __m128d imags = _mm_load_pd(&cosines[n + sin]);
    return _mm256_set_m128d(_mm_unpackhi_pd(reals, imags),
                            _mm_unpacklo_pd(reals, imags));
  }

  static __m256d mul(__m256d a, __m256d b) {
    return _mm256_fmaddsub_pd(
        _mm256_movedup_pd(a), b,
        _mm256_mul_pd(_mm256_permute_pd(a, 15), _mm256_permute_pd(b, 5)));
  }

  static constexpr int FFT_CUTOFF = 14;
  static constexpr int CT8_CUTOFF = 15;
  static constexpr int FFT_MIN =
      FFT_CUTOFF - 1; // -1 due to real-fft size halving optimization
  static constexpr int FFT_MAX = 20;

  using Complex = double[2];

  static void fft_cooley_tukey_no_transpose_4(Complex *data, int n_pow) {
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

  static void fft_cooley_tukey_no_transpose_8(Complex *data, int n_pow,
                                              int count3) {
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

  static constexpr std::pair<int, int> get_counts(int n_pow) {
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

  static uint64_t shiftl(uint64_t x, int shift) {
    if (shift > 0) {
      return x << shift;
    } else {
      return x >> (-shift);
    }
  }
  __attribute__((always_inline)) static __m256i shiftl(__m256i x, int shift) {
    if (shift > 0) {
      return _mm256_slli_epi64(x, shift);
    } else {
      return _mm256_srli_epi64(x, -shift);
    }
  }

  template <int N_POW>
  static uint64_t reverse_mixed_radix_const64(uint64_t number) {
    static constexpr int COUNT3 = get_counts(N_POW).first;
    static constexpr int COUNT2 = get_counts(N_POW).second;
    uint64_t result = 0;
    for (int i = 0; i < COUNT3; i++) {
      result |=
          shiftl(number & (uint64_t{7} << (i * 3)), N_POW - (i * 2 + 1) * 3);
    }
    for (int i = 0; i < COUNT2; i++) {
      result |= shiftl(number & (uint64_t{3} << (COUNT3 * 3 + i * 2)),
                       N_POW - COUNT3 * 3 * 2 - (i * 2 + 1) * 2);
    }
    return result;
  }
#ifndef __clang__
#pragma GCC push_options
#pragma GCC optimize("O2")
#endif
  template <int N_POW, int... Iterator3, int... Iterator2>
  __attribute__((always_inline)) static __m256i
  reverse_mixed_radix_const256_impl(__m256i number,
                                    std::integer_sequence<int, Iterator3...>,
                                    std::integer_sequence<int, Iterator2...>) {
    static constexpr int COUNT3 = get_counts(N_POW).first;
    // Trick the compiler into believing the state of ymm{1..7} has to be
    // preserved during the function execution so that we are free to remove
    // these registers from clobber list when dynamically calling this function
    register __m256d ymm1 asm("ymm1");
    register __m256d ymm2 asm("ymm2");
    register __m256d ymm3 asm("ymm3");
    register __m256d ymm4 asm("ymm4");
    register __m256d ymm5 asm("ymm5");
    register __m256d ymm6 asm("ymm6");
    register __m256d ymm7 asm("ymm7");
    // Save the state of the registers at the beginning of the function
    asm volatile(""
                 : "=x"(ymm1), "=x"(ymm2), "=x"(ymm3), "=x"(ymm4), "=x"(ymm5),
                   "=x"(ymm6), "=x"(ymm7));
    __m256i result = _mm256_setzero_si256();
    ((result = _mm256_or_si256(
          result,
          shiftl(_mm256_and_si256(number, _mm256_set1_epi64x(
                                              uint64_t{7} << (Iterator3 * 3))),
                 N_POW - (Iterator3 * 2 + 1) * 3))),
     ...);
    ((result = _mm256_or_si256(
          result,
          shiftl(_mm256_and_si256(
                     number, _mm256_set1_epi64x(
                                 uint64_t{3} << (COUNT3 * 3 + Iterator2 * 2))),
                 N_POW - COUNT3 * 3 * 2 - (Iterator2 * 2 + 1) * 2))),
     ...);
    // Restore the state of the registers. Prevent reordering of `asm volatile`
    // with computation of the result by specifying the latter as an input
    asm volatile(""
                 :
                 : "x"(result), "x"(ymm1), "x"(ymm2), "x"(ymm3), "x"(ymm4),
                   "x"(ymm5), "x"(ymm6), "x"(ymm7));
    return result;
  }
  template <int N_POW>
  static __m256i reverse_mixed_radix_const256(__m256i number) {
    static constexpr int COUNT3 = get_counts(N_POW).first;
    static constexpr int COUNT2 = get_counts(N_POW).second;
    return reverse_mixed_radix_const256_impl<N_POW>(
        number, std::make_integer_sequence<int, COUNT3>(),
        std::make_integer_sequence<int, COUNT2>());
  }
#ifndef __clang__
#pragma GCC pop_options
#endif

  template <int... Pows>
  static uint64_t reverse_mixed_radix_dyn(int n_pow, uint64_t number,
                                          std::integer_sequence<int, Pows...>) {
    static constexpr uint64_t (*dispatch[])(uint64_t) = {
        &reverse_mixed_radix_const64<FFT_MIN + Pows>...};
    return dispatch[n_pow - FFT_MIN](number);
  }
  template <int... Pows>
  __attribute__((always_inline)) static __m256i
  reverse_mixed_radix_dyn(int n_pow, __m256i vec,
                          std::integer_sequence<int, Pows...>) {
    static constexpr __m256i (*dispatch[])(__m256i) = {
        &reverse_mixed_radix_const256<FFT_MIN + Pows>...};
    register __m256i ymm0 asm("ymm0") = vec;
    asm volatile("call *%[addr];"
                 : "+x"(ymm0)
                 : [addr] "m"(dispatch[n_pow - FFT_MIN])
                 : "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm13",
                   "ymm14", "ymm15");
    return ymm0;
  }

  static uint64_t reverse_mixed_radix_dyn(int n_pow, uint64_t number) {
    return reverse_mixed_radix_dyn(
        n_pow, number,
        std::make_integer_sequence<int, FFT_MAX - FFT_MIN + 1>());
  }
  static std::array<uint64_t, 4> reverse_mixed_radix_dyn(int n_pow, uint64_t a,
                                                         uint64_t b, uint64_t c,
                                                         uint64_t d) {
    // This should be autovectorized
    auto res = reverse_mixed_radix_dyn(
        n_pow,
        _mm256_set_epi64x(static_cast<int64_t>(d), static_cast<int64_t>(c),
                          static_cast<int64_t>(b), static_cast<int64_t>(a)),
        std::make_integer_sequence<int, FFT_MAX - FFT_MIN + 1>());
    return {static_cast<uint64_t>(_mm256_extract_epi64(res, 0)),
            static_cast<uint64_t>(_mm256_extract_epi64(res, 1)),
            static_cast<uint64_t>(_mm256_extract_epi64(res, 2)),
            static_cast<uint64_t>(_mm256_extract_epi64(res, 3))};
  }

  static auto fft_cooley_tukey(Complex *data, int n_pow) {
    ensure_twiddle_factors(n_pow);
    int count3 = get_counts(n_pow).first;
    fft_cooley_tukey_no_transpose_8(data, n_pow, count3);
    return [n_pow](auto... args) {
      return reverse_mixed_radix_dyn(n_pow, args...);
    };
  }

  static int get_fft_n_pow(const BigInt &lhs, const BigInt &rhs) {
    return 64 -
           __builtin_clzll((lhs.data.size() + rhs.data.size() - 1) * 4 - 1);
  }

  static inline std::vector<double> cosines{0};
  static void ensure_twiddle_factors(int want_n_pow) {
    while (cosines.size() < (size_t{2} << want_n_pow)) {
      size_t n = cosines.size();
      cosines.resize(2 * n);
      double coeff = 2 * M_PI / static_cast<double>(n);
      for (size_t k = 0; k < n / 2; k++) {
        double c = std::cos(coeff * static_cast<double>(k));
        cosines[n + k] = c;
        cosines[n + n / 2 + k] = -c;
      }
    }
  }

  // Credits to https://stackoverflow.com/a/41148578
  // Only work for inputs in the range: [0, 2^52)
  static __m256i double_to_uint64(__m256d x) {
    x = _mm256_add_pd(x, _mm256_set1_pd(0x0010000000000000));
    return _mm256_castpd_si256(
        _mm256_xor_pd(x, _mm256_set1_pd(0x0010000000000000)));
  }
  static __m256d uint64_to_double(__m256i x) {
    auto y = _mm256_castsi256_pd(x);
    y = _mm256_or_pd(y, _mm256_set1_pd(0x0010000000000000));
    return _mm256_sub_pd(y, _mm256_set1_pd(0x0010000000000000));
  }

  static BigInt mul_fft(const BigInt &lhs, const BigInt &rhs) {
    int n_pow = get_fft_n_pow(lhs, rhs);
    size_t n = size_t{1} << n_pow;

    // Split numbers into words
    Complex *united_fft = new (std::align_val_t(32)) Complex[n];
    std::memset(united_fft, 0, sizeof(Complex) * n);

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
    size_t united_fft_one = united_fft_access(uint64_t{1});
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
      auto [ani0a, ani1a, ani0a4, ani1a4] =
          united_fft_access(ni0a, ni1a, ni0a - 4, ni1a - 4);
      auto [ani0b, ani1b, ani0b4, ani1b4] =
          united_fft_access(ni0b, ni1b, ni0b - 4, ni1b - 4);
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
      __m256d f = _mm256_fmaddsub_pd(w_real0011, d,
                                     _mm256_fmaddsub_pd(w_imag0011, e, c));
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
      __m256d error =
          _mm256_andnot_pd(_mm256_set1_pd(-0.),
                           _mm256_sub_pd(fp_value, uint64_to_double(value)));
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
    assert(max_error < 0.4);

    if (carry > 0) {
      result.data.push_back(carry);
    } else {
      while (result.data.back() == 0) {
        result.data.pop_back();
      }
    }

    // std::cerr << lhs.data.size() + rhs.data.size() << " -> " <<
    // result.data.size() << std::endl;
    assert(result.data.size() == lhs.data.size() + rhs.data.size() ||
           result.data.size() == lhs.data.size() + rhs.data.size() - 1);

    return result;
  }

public:
  BigInt() {}

  BigInt(__uint128_t value) {
    data = {static_cast<uint64_t>(value), static_cast<uint64_t>(value >> 64)};
    data.set_size((value > 0) + (value >= (__uint128_t{1} << 64)));
  }
  BigInt(uint64_t value) {
    data = {value};
    data.set_size(value > 0);
  }
  BigInt(int value) {
    assert(value >= 0);
    if (value > 0) {
      data = {static_cast<uint64_t>(value)};
    }
  }

  template <typename List, typename = decltype(std::declval<List>().begin())>
  BigInt(List &&list, with_base base) {
    *this = str_to_int(list.begin(), list.end(), base.base,
                       [](uint64_t digit) { return digit; });
  }
  BigInt(std::string_view s, with_base base = {10}) {
    assert(base.base <= 36);
    *this = str_to_int(s.rbegin(), s.rend(), base.base, [base](char c) {
      uint64_t digit;
      if ('0' <= c && c <= '9') {
        digit = static_cast<uint64_t>(c - '0');
      } else if ('a' <= c && c <= 'z') {
        digit = static_cast<uint64_t>(c - 'a' + 10);
      } else {
        assert(false);
      }
      assert(digit < base.base);
      return digit;
    });
  }
  BigInt(const char *s, with_base base = {10})
      : BigInt(std::string_view(s), base) {}

  BigInt(const BigInt &rhs) : data(rhs.data) {}
  BigInt(BigInt &&rhs) : data(std::move(rhs.data)) {}

  BigInt &operator=(const BigInt &rhs) {
    data = rhs.data;
    return *this;
  }

  BigInt &operator=(BigInt &&rhs) {
    data = std::move(rhs.data);
    return *this;
  }

  bool operator==(const BigInt &rhs) const { return data == rhs.data; }
  bool operator!=(const BigInt &rhs) const { return !(*this == rhs); }

  bool operator<(const BigInt &rhs) const {
    if (data.size() != rhs.data.size()) {
      return data.size() < rhs.data.size();
    }
    return std::lexicographical_compare(data.rbegin(), data.rend(),
                                        rhs.data.rbegin(), rhs.data.rend());
  }
  bool operator>(const BigInt &rhs) const { return rhs < *this; }
  bool operator<=(const BigInt &rhs) const { return !(rhs < *this); }
  bool operator>=(const BigInt &rhs) const { return !(*this < rhs); }

  BigInt &operator+=(const BigInt &rhs) {
    if (__builtin_expect(data.empty(), 0)) {
      return *this = rhs;
    } else if (__builtin_expect(rhs.data.empty(), 0)) {
      return *this;
    }

    size_t i = 0;
    uint64_t value = 0;
    size_t loop_count = std::min(data.size(), rhs.data.size());
    asm volatile(
        "clc;"
        "1:"
        "mov (%[rhs_data_ptr],%[i],8), %[value];"
        "adc %[value], (%[data_ptr],%[i],8);"
        "inc %[i];"
        "dec %[loop_count];"
        "jnz 1b;"
        "mov $0, %[value];"
        "adc $0, %[value];"
        : [i] "+r"(i), [value] "+r"(value), [loop_count] "+r"(loop_count)
        : [data_ptr] "r"(data.data()), [rhs_data_ptr] "r"(rhs.data.data())
        : "flags", "memory");

    if (data.size() < rhs.data.size()) {
      data.increase_size(rhs.data.size());
      if (value) {
        while (i < rhs.data.size() &&
               rhs.data[i] == static_cast<uint64_t>(-1)) {
          data[i] = 0;
          i++;
        }
        if (__builtin_expect(i == rhs.data.size(), 0)) {
          data.push_back(1);
        } else {
          data[i] = rhs.data[i] + 1;
          i++;
        }
      }
      std::copy(rhs.data.begin() + i, rhs.data.end(), data.begin() + i);
    } else {
      if (value) {
        for (; i < data.size(); i++) {
          data[i]++;
          if (data[i] != 0) {
            break;
          }
        }
        if (__builtin_expect(i == data.size(), 0)) {
          data.push_back(1);
        }
      }
    }
    return *this;
  }

  BigInt &operator-=(const BigInt &rhs) {
    if (rhs.data.empty()) {
      return *this;
    }

    assert(data.size() >= rhs.data.size());

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

    while (!data.empty() && data.back() == 0) {
      data.pop_back();
    }

    return *this;
  }

  BigInt &operator++() { return *this += 1; }
  BigInt operator++(int) {
    BigInt tmp = *this;
    ++*this;
    return tmp;
  }

  BigInt operator+(const BigInt &rhs) const & {
    BigInt tmp = *this;
    tmp += rhs;
    return tmp;
  }
  BigInt operator+(BigInt &&rhs) const & { return std::move(rhs += *this); }
  BigInt operator+(const BigInt &rhs) && { return std::move(*this += rhs); }
  BigInt operator+(BigInt &&rhs) && { return std::move(*this += rhs); }

  BigInt &operator--() { return *this -= 1; }
  BigInt operator--(int) {
    BigInt tmp = *this;
    --*this;
    return tmp;
  }

  BigInt operator-(const BigInt &rhs) const & {
    BigInt tmp = *this;
    tmp -= rhs;
    return tmp;
  }
  BigInt operator-(const BigInt &rhs) && { return std::move(*this -= rhs); }

  BigInt &operator*=(uint64_t rhs) {
    if (rhs == 0) {
      data.clear_dealloc();
      return *this;
    }
    uint64_t carry = 0;
    for (size_t i = 0; i < data.size(); i++) {
      __uint128_t total = __uint128_t{data[i]} * rhs + carry;
      data[i] = static_cast<uint64_t>(total);
      carry = static_cast<uint64_t>(total >> 64);
    }
    if (carry != 0) {
      data.push_back(carry);
    }
    return *this;
  }

  BigInt operator*(uint64_t rhs) const & {
    BigInt tmp = *this;
    tmp *= rhs;
    return tmp;
  }
  BigInt operator*(uint64_t rhs) && { return std::move(*this *= rhs); }

  BigInt operator*(const BigInt &rhs) const {
    if (data.empty() || rhs.data.empty()) {
      return {};
    }
    int n_pow = get_fft_n_pow(*this, rhs);
    if (n_pow >= FFT_CUTOFF) {
      return mul_fft(*this, rhs);
    }
    BigInt result;
    result.data.increase_size_zerofill(data.size() + rhs.data.size());
    mul_at(result, *this, rhs, 0);
    while (result.data.back() == 0) {
      result.data.pop_back();
    }
    return result;
  }

  friend std::ostream &operator<<(std::ostream &out, const BigInt &rhs) {
    if (rhs.data.empty()) {
      return out << "0x0";
    }
    out << "0x" << std::hex << rhs.data.back() << std::setfill('0');
    for (auto it = rhs.data.rbegin() + 1; it != rhs.data.rend(); ++it) {
      out << std::setw(16) << *it;
    }
    return out << std::dec << std::setfill(' ');
  }
};

} // namespace bigint

int main() {
  using Int = bigint::BigInt;

  std::srand(1);

  std::string s;
  for (int i = 0; i < 1200000; i++) {
    s.push_back(static_cast<char>('0' + rand() % 2));
  }

  // mpz_class a(s);
  // mpz_class b = a;
  // clock_t start = clock();
  Int a = {s.c_str(), bigint::with_base{10}};

  // std::cerr << a.data.size() << std::endl;

  clock_t start = clock();
  for (int i = 0; i < 100; i++) {
    // (mpz_class)(a * b);
    (a * a);
    // ((a * a) * (a * a)) * ((a * a) * (a * a));
  }
  std::cerr << static_cast<double>(clock() - start) / CLOCKS_PER_SEC
            << std::endl;

  // std::cerr << std::endl;
  // std::cout << a << std::endl;
  // std::cout << s << std::endl;

  return 0;
}

#include <algorithm>
#include <array>
#include <cassert>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <tuple>
#include <vector>

// #define class struct

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

// constexpr std::pair<uint64_t, int>
// virtual_base_to_physical(uint64_t virtual_base) {
//   uint64_t physical_base = 1;
//   int multiplier = 0;
//   while (physical_base <= SIZE_MAX / virtual_base) {
//     physical_base *= virtual_base;
//     multiplier++;
//   }
//   return {physical_base, multiplier};
// }

// template <uint64_t VirtualBase> class BaseMixin {
//   static constexpr uint64_t PHYSICAL_BASE =
//       virtual_base_to_physical(VirtualBase).first;

// public:
//   constexpr uint64_t get_virtual_base() const { return VirtualBase; }
//   constexpr uint64_t get_physical_base() const { return PHYSICAL_BASE; }
//   constexpr uint64_t get_multiplier() const {
//     return virtual_base_to_physical(VirtualBase).second;
//   }

//   constexpr std::pair<uint64_t, uint64_t> phy_divmod(uint64_t value) const {
//     return {value / PHYSICAL_BASE, value % PHYSICAL_BASE};
//   }
//   constexpr std::pair<uint64_t, bool> phy_add_carry(uint64_t a, uint64_t b,
//                                                     bool carry) const {
//     if constexpr (PHYSICAL_BASE <= 0x8000000000000000ULL) {
//       uint64_t sum = a + b + carry;
//       if (sum >= PHYSICAL_BASE) {
//         return {sum - PHYSICAL_BASE, 1};
//       } else {
//         return {sum, 0};
//       }
//     } else {
//       unsigned long long carry1 = 0;
//       uint64_t sum = __builtin_addcll(a, b, carry, &carry1);
//       if (sum >= PHYSICAL_BASE) {
//         carry1 = 1;
//       }
//       if (carry1) {
//         sum -= PHYSICAL_BASE;
//       }
//       return {sum, carry1};
//     }
//   }

//   constexpr std::pair<uint64_t, uint64_t> virt_divmod(uint64_t value) const {
//     return {value / get_virtual_base(), value % get_virtual_base()};
//   }

//   constexpr bool operator==(const BaseMixin &rhs) const { return true; }
// };

// template <> class BaseMixin<0> {
//   uint64_t virtual_base;
//   uint64_t physical_base;
//   uint64_t multiplier;
//   libdivide::divisor virt_divisor;
//   libdivide::divisor phy_divisor;

// public:
//   BaseMixin(uint64_t virtual_base) : virtual_base(virtual_base) {
//     std::tie(physical_base, multiplier) =
//         virtual_base_to_physical(virtual_base);
//     virt_divisor = libdivide::divisor{virtual_base};
//     phy_divisor = libdivide::divisor{physical_base};
//   }

//   uint64_t get_virtual_base() const { return virtual_base; }
//   uint64_t get_physical_base() const { return physical_base; }
//   uint64_t get_multiplier() const { return multiplier; }

//   std::pair<uint64_t, uint64_t> phy_divmod(uint64_t value) const {
//     uint64_t div = value / phy_divisor;
//     uint64_t mod = value - div * physical_base;
//     return {div, mod};
//   }
//   std::pair<uint64_t, bool> phy_add_carry(uint64_t a, uint64_t b,
//                                           bool carry) const {
//     unsigned long long carry1;
//     uint64_t sum = __builtin_addcll(a, b, carry, &carry1);
//     if (sum >= get_physical_base()) {
//       carry1 = 1;
//     }
//     if (carry1) {
//       sum -= get_physical_base();
//     }
//     return {sum, carry1};
//   }

//   std::pair<uint64_t, uint64_t> virt_divmod(uint64_t value) const {
//     uint64_t div = value / virt_divisor;
//     uint64_t mod = value - div * virtual_base;
//     return {div, mod};
//   }

//   bool operator==(const BaseMixin &rhs) const {
//     return virtual_base == rhs.virtual_base;
//   }
// };

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
  SmallVec() : _begin(_inline_storage), _size(0), _capacity(INLINE_STORAGE_SIZE) {}
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
  // Stores 32-bit numbers
  SmallVec data;

  template <typename Iterator, typename Map>
  static BigInt str_to_int(Iterator begin, Iterator end, uint64_t base, Map map,
                           const BigInt *powers_of_base, int max_block_len) {
    if (end - begin <= max_block_len) {
      uint64_t val = 0;
      for (auto it = end; it != begin;) {
        val *= base;
        val += map(*--it);
      }
      return val;
    }

    int low_len_pow = 63 - __builtin_clzll(end - begin - 1);
    uint64_t low_len = uint64_t{1} << low_len_pow;
    Iterator mid = begin + low_len;
    BigInt low =
        str_to_int(begin, mid, base, map, powers_of_base, max_block_len);
    BigInt high =
        str_to_int(mid, end, base, map, powers_of_base, max_block_len);
    return high * powers_of_base[low_len_pow] + low;
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
    while ((uint64_t{1} << powers_of_base.size()) <= end - begin) {
      powers_of_base.push_back(powers_of_base.back() * powers_of_base.back());
    }

    return str_to_int(begin, end, base, map, powers_of_base.data(),
                      max_block_len);
  }

  static void add_at(BigInt &lhs, const BigInt &rhs, size_t offset) {
    if (__builtin_expect(rhs.data.empty(), 0)) {
      return;
    }

    uint64_t *ldata = lhs.data.data() + offset;
    size_t lsize = lhs.data.size() - offset;

    size_t i = 0;
    uint64_t value = 0;
    size_t loop_count = std::min(lsize, rhs.data.size());
    asm("clc;"
        "1:"
        "mov (%[rhs_data_ptr],%[i],8), %[value];"
        "adc %[value], (%[data_ptr],%[i],8);"
        "inc %[i];"
        "dec %[loop_count];"
        "jnz 1b;"
        "mov $0, %[value];"
        "adc $0, %[value];"
        : [i] "+r"(i), [value] "+r"(value), [loop_count] "+r"(loop_count)
        : [data_ptr] "r"(ldata), [rhs_data_ptr] "r"(rhs.data.data())
        : "flags", "memory");

    if (lsize < rhs.data.size()) {
      lhs.data.increase_size(rhs.data.size());
      ldata = lhs.data.data() + offset;
      if (value) {
        while (i < rhs.data.size() &&
               rhs.data[i] == static_cast<uint64_t>(-1)) {
          ldata[i] = 0;
          i++;
        }
        if (__builtin_expect(i == rhs.data.size(), 0)) {
          lhs.data.push_back(1);
        } else {
          ldata[i] = rhs.data[i] + 1;
          i++;
        }
      }
      std::copy(rhs.data.begin() + i, rhs.data.end(), ldata + i);
    } else {
      if (value) {
        for (; i < lsize; i++) {
          ldata[offset]++;
          if (ldata[offset] != 0) {
            break;
          }
        }
        if (__builtin_expect(i == lsize, 0)) {
          lhs.data.push_back(1);
        }
      }
    }
  }

  static BigInt mul_1x1(const BigInt &lhs, const BigInt &rhs) {
    return __uint128_t{lhs.data[0]} * rhs.data[0];
  }

  static BigInt mul_quadratic(const BigInt &lhs, const BigInt &rhs) {
    BigInt res;
    res.data.increase_size_zerofill(lhs.data.size() + rhs.data.size() - 1);

    uint64_t carry_low = 0;
    uint64_t carry_high = 0;
    for (size_t i = 0; i < res.data.size(); i++) {
      size_t left =
          std::max(static_cast<ssize_t>(i + 1 - rhs.data.size()), ssize_t{0});
      size_t right = std::min(i + 1, lhs.data.size());

      uint64_t sum_low = carry_low;
      uint64_t sum_mid = carry_high;
      uint64_t sum_high = 0;
      while (left < right) {
        uint64_t rax = lhs.data[left];
        asm("mulq %[b];"
            "add %%rax, %[sum_low];"
            "adc %%rdx, %[sum_mid];"
            "adc $0, %[sum_high];"
            : "+a"(rax), [sum_low] "+r"(sum_low), [sum_mid] "+r"(sum_mid),
              [sum_high] "+r"(sum_high)
            : [b] "m"(rhs.data[i - left])
            : "flags", "rdx");
        left++;
      }

      res.data[i] = sum_low;
      carry_low = sum_mid;
      carry_high = sum_high;
    }

    if (carry_high > 0) {
      res.data.push_back(carry_low);
      res.data.push_back(carry_high);
    } else if (carry_low > 0) {
      res.data.push_back(carry_low);
    }

    return res;
  }

  static BigInt mul_quadratic_simd(const BigInt &lhs, const BigInt &rhs) {
    BigInt res;
    res.data.increase_size_zerofill(lhs.data.size() + rhs.data.size());

    const uint32_t *ldata = reinterpret_cast<const uint32_t *>(lhs.data.data());
    const uint32_t *rdata = reinterpret_cast<const uint32_t *>(rhs.data.data());
    uint32_t *data = reinterpret_cast<uint32_t *>(res.data.data());

    int64_t carry = 0;
    for (size_t i = 0; i < res.data.size() * 2; i++) {
      size_t left = std::max(static_cast<ssize_t>(i + 1 - rhs.data.size() * 2),
                             ssize_t{0});
      size_t right = std::min(i + 1, lhs.data.size() * 2);

      // sum_low is signed
      __m256i sum_low = _mm256_set_epi64x(carry, 0, 0, 0);
      __m256i sum_high = _mm256_setzero_si256();

      while (left + 8 <= right) {
#define X                                                                      \
  do {                                                                         \
    __m128i lhs32 =                                                            \
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(ldata + left));      \
    __m128i rhs32 = _mm_loadu_si128(                                           \
        reinterpret_cast<const __m128i *>(rdata + i - left - 3));              \
    rhs32 = _mm_shuffle_epi32(rhs32, 0b00011011);                              \
    __m256i lhs64 = _mm256_cvtepu32_epi64(lhs32);                              \
    __m256i rhs64 = _mm256_cvtepu32_epi64(rhs32);                              \
    __m256i product = _mm256_mul_epu32(lhs64, rhs64);                          \
    __m256i sum_low_new = _mm256_add_epi64(sum_low, product);                  \
    __m256i overflow = _mm256_cmpgt_epi64(sum_low, sum_low_new);               \
    sum_low = sum_low_new;                                                     \
    sum_high = _mm256_sub_epi64(sum_high, overflow);                           \
    left += 4;                                                                 \
  } while (0);
        X X
#undef X
      }

      // Convert sum_low to unsigned
      __m256i negative = _mm256_cmpgt_epi64(_mm256_setzero_si256(), sum_low);
      sum_high = _mm256_add_epi64(sum_high, negative);

      // Horizontal sum
      uint64_t sum_low_val = 0;
      uint64_t sum_high_val = 0;

#define X(i)                                                                   \
  do {                                                                         \
    uint64_t word = _mm256_extract_epi64(sum_low, i);                          \
    asm("add %[word], %[sum_low_val];"                                         \
        "adc $0, %[sum_high_val];"                                             \
        : [sum_low_val] "+r"(sum_low_val), [sum_high_val] "+r"(sum_high_val)   \
        : [word] "r"(word)                                                     \
        : "flags");                                                            \
    sum_high_val += _mm256_extract_epi64(sum_high, i);                         \
  } while (0);
      X(0)
      X(1)
      X(2)
      X(3)
#undef X

      while (left < right) {
        asm("add %[product], %[sum_low_val];"
            "adc $0, %[sum_high_val];"
            : [sum_low_val] "+r"(sum_low_val), [sum_high_val] "+r"(sum_high_val)
            : [product] "r"(uint64_t{ldata[left]} * rdata[i - left])
            : "flags");
        left++;
      }

      data[i] = static_cast<uint32_t>(sum_low_val);
      carry = (sum_high_val << 32) | (sum_low_val >> 32);
    }

    if (carry > 0) {
      res.data.push_back(carry);
    } else if (res.data.back() == 0) {
      res.data.pop_back();
    }

    return res;
  }

  static BigInt mul_karatsuba(const BigInt &lhs, const BigInt &rhs) {
    size_t b = std::min(lhs.data.size(), rhs.data.size()) / 2;

    BigInt x0, x1, y0, y1;
    x0.data = {lhs.data.data(), b};
    x1.data = {lhs.data.data() + b, lhs.data.size() - b};
    y0.data = {rhs.data.data(), b};
    y1.data = {rhs.data.data() + b, rhs.data.size() - b};
    while (!x0.data.empty() && x0.data.back() == 0) {
      x0.data.pop_back();
    }
    while (!y0.data.empty() && y0.data.back() == 0) {
      y0.data.pop_back();
    }

    BigInt z0 = x0 * y0;
    BigInt z2 = x1 * y1;
    BigInt z1 = (x0 + x1) * (y0 + y1) - z0 - z2;
    BigInt result;
    result.data.increase_size(b * 2 + z2.data.size());
    std::copy(z0.data.begin(), z0.data.end(), result.data.begin());
    std::fill(result.data.begin() + z0.data.size(), result.data.begin() + b * 2,
              0);
    std::copy(z2.data.begin(), z2.data.end(), result.data.begin() + b * 2);
    add_at(result, z1, b);
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
      int digit;
      if ('0' <= c && c <= '9') {
        digit = c - '0';
      } else if ('a' <= c && c <= 'z') {
        digit = c - 'a' + 10;
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
      *this = rhs;
    } else {
      add_at(*this, rhs, 0);
    }
    return *this;
  }

  BigInt &operator-=(const BigInt &rhs) {
    if (rhs.data.empty()) {
      return *this;
    }

    assert(data.size() >= rhs.data.size());

    size_t i = 0;
    uint64_t value = 0;
    size_t loop_count = rhs.data.size();
    asm("clc;"
        "1:"
        "mov (%[rhs_data_ptr],%[i],8), %[value];"
        "sbb %[value], (%[data_ptr],%[i],8);"
        "inc %[i];"
        "dec %[loop_count];"
        "jnz 1b;"
        "jnc 2f;"
        "3:"
        "subq $1, (%[data_ptr],%[i],8);"
        "inc %[i];"
        "jc 3b;"
        "2:"
        : [i] "+r"(i), [value] "+r"(value), [loop_count] "+r"(loop_count)
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
      carry = total >> 64;
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
    const BigInt *l = this;
    const BigInt *r = &rhs;
    if (l->data.size() < r->data.size()) {
      std::swap(l, r);
    }
    if (l->data.size() == 1) {
      return mul_1x1(*l, *r);
    }
    // if (std::min(l->data.size(), r->data.size()) >= 384) {
    //   return mul_toom3(*l, *r);
    // } else
    if (std::min(l->data.size(), r->data.size()) >= 128) {
      return mul_karatsuba(*l, *r);
      // } else if (std::min(l->data.size(), r->data.size()) >= 8) {
      //   return mul_quadratic_simd(*l, *r);
    } else {
      return mul_quadratic(*l, *r);
    }
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

namespace base64 {

uint8_t parse_char(char symbol) {
  if (symbol >= 48 && symbol <= 57)
    return symbol - 48;
  if (symbol >= 65 && symbol <= 90)
    return symbol - 55;
  if (symbol >= 97 && symbol <= 122)
    return symbol - 61;
  if (symbol == 32)
    return 62;
  if (symbol == 46)
    return 63;
  return 64;
}

bigint::BigInt parse_message(const std::string &s) {
  std::vector<uint8_t> chars(s.size());
  for (size_t i = 0; i < s.size(); i++) {
    chars[i] = parse_char(s[i]);
  }
  return {chars, bigint::with_base{64}};
}

} // namespace base64

int main() {
  // uint32_t p, g, k;
  // std::cin >> p >> g >> k;

  // std::string message;
  // std::getline(std::cin, message);
  // std::getline(std::cin, message);

  // auto msg = base64::parse_message(message);
  // std::cout << msg << std::endl;

  using Int = bigint::BigInt;

  // Int a{};

  // Int a = 1;
  // a += 2;

  std::string s;
  for (int i = 0; i < 1000000; i++) {
    s.push_back('0' + rand() % 10);
  }

  // std::cout << s.size() << std::endl;

  Int a = {s.c_str(), bigint::with_base{16}};

  // int start = clock();
  // for(size_t i = 0; i < 1000000; i++) {
  //   a + a;
  // }
  // std::cerr << (float)(clock() - start) / CLOCKS_PER_SEC << std::endl;

  // std::cout << a << std::endl;
  // std::cout << "0x" << s << std::endl;

  return 0;
}

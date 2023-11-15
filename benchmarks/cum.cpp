#include <lib.hpp>
#include <cum.cpp>
#include <gmpxx.h>
#include <random>

using namespace bigint;

template<typename F>
void measure(F f) {
  size_t count = 1;
  for (;;) {
    clock_t start = clock();
    for (size_t i = 0; i < count; i++) {
      f();
    }
    if (clock() - start >= CLOCKS_PER_SEC / 5) {
      break;
    }
    count *= 2;
  }
  clock_t start = clock();
  for (size_t i = 0; i < count * 2; i++) {
    f();
  }
  double secs_per_iteration = static_cast<double>(clock() - start) / CLOCKS_PER_SEC / (count * 2);
  std::cout << secs_per_iteration * 1e9;
  if (secs_per_iteration >= 1) {
    std::cerr << secs_per_iteration << " s/iteration";
  } else if (secs_per_iteration >= 1e-3) {
    std::cerr << secs_per_iteration * 1e3 << " ms/iteration";
  } else if (secs_per_iteration >= 1e-6) {
    std::cerr << secs_per_iteration * 1e6 << " us/iteration";
  } else {
    std::cerr << secs_per_iteration * 1e9 << " ns/iteration";
  }
}

template<typename F>
void with_sizes(F f) {
  for (size_t base = 4; base <= 40000000; base *= 2) {
    for (size_t bits = 10 * base; bits < 20 * base; bits += base) {
      f(bits);
    }
  }
}

int main() {
  std::cerr << std::setprecision(3);

  gmp_randclass rng(gmp_randinit_default);
  rng.seed(1);

  FFT fft;

  with_sizes([&](size_t bits) {
    mpz_class a_gmp = rng.get_z_bits(bits);
    mpz_class b_gmp = rng.get_z_bits(bits);

    BigInt a_xcrypto, b_xcrypto;
    a_xcrypto.data = SmallVec{a_gmp.get_mpz_t()->_mp_d, static_cast<size_t>(a_gmp.get_mpz_t()->_mp_size)};
    b_xcrypto.data = SmallVec{b_gmp.get_mpz_t()->_mp_d, static_cast<size_t>(b_gmp.get_mpz_t()->_mp_size)};

    u32* ptr = reinterpret_cast<u32*>(a_gmp.get_mpz_t()->_mp_d);
    std::vector<u32> a_cum(ptr, ptr + a_gmp.get_mpz_t()->_mp_size * 2);
    ptr = reinterpret_cast<u32*>(b_gmp.get_mpz_t()->_mp_d);
    std::vector<u32> b_cum(ptr, ptr + b_gmp.get_mpz_t()->_mp_size * 2);

    std::cerr << bits << " bits: libgmp: ";
    std::cout << bits << '\t';
    measure([&]() {
      (mpz_class)(a_gmp * b_gmp);
    });
    std::cerr << ", xcrypto: ";
    std::cout << '\t';
    measure([&]() {
      a_xcrypto * b_xcrypto;
    });
    std::cerr << ", cum: ";
    std::cout << '\t';
    measure([&]() {
      fft.convolve(a_cum, b_cum);
    });
    std::cerr << std::endl;
    std::cout << std::endl;
  });

  return 0;
}

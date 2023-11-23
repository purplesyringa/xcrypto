#include <gmpxx.h>
#include <lib.hpp>
#include "measure.hpp"
#include <random>

using namespace bigint;

int main() {
  std::cerr << std::setprecision(3);

  gmp_randclass rng(gmp_randinit_default);
  rng.seed(1);

  with_sizes([&](size_t bits) {
    mpz_class a_gmp = rng.get_z_bits(bits);
    mpz_class b_gmp = rng.get_z_bits(bits);

    BigInt a_xcrypto, b_xcrypto;
    a_xcrypto.data = SmallVec{a_gmp.get_mpz_t()->_mp_d, static_cast<size_t>(a_gmp.get_mpz_t()->_mp_size)};
    b_xcrypto.data = SmallVec{b_gmp.get_mpz_t()->_mp_d, static_cast<size_t>(b_gmp.get_mpz_t()->_mp_size)};

    std::cerr << bits << " bits: GMP: ";
    std::cout << bits << '\t';
    measure([&]() {
      (mpz_class)(a_gmp + b_gmp);
    });
    std::cerr << ", xcrypto: ";
    std::cout << '\t';
    measure([&]() {
      a_xcrypto + b_xcrypto;
    });
    std::cerr << std::endl;
    std::cout << std::endl;
  });

  return 0;
}

#include "lib.hpp"

int main() {
  using Int = bigint::BigInt;

  std::srand(1);

  std::string s;
  for (int i = 0; i < 50000000; i++) {
    s.push_back(static_cast<char>('0' + rand() % 10));
  }

  // mpz_class a(s);
  // mpz_class b = a;
  clock_t start = clock();
  Int a = {s.c_str(), bigint::with_base{10}};

  // while (true) {
  //   Int b = a;
  //   if (b.divmod_inplace(3) == 0) {
  //     break;
  //   }
  //   a++;
  // }

  std::cerr << a.data.size() << std::endl;

  // clock_t start = clock();
  // for (int i = 0; i < 10000; i++) {
  // (mpz_class)(a * b);
  // (a * a);
  // ((a * a) * (a * a)) * ((a * a) * (a * a));
  // Int b = a;
  // b.halve();
  // b.divide_inplace_whole(3);
  // c.divmod_inplace(3);
  // std::cerr << b << std::endl;
  // return 0;
  // }
  std::cerr << static_cast<double>(clock() - start) / CLOCKS_PER_SEC
            << std::endl;

  // std::cerr << std::endl;
  // std::cout << a << std::endl;
  // std::cout << s << std::endl;

  return 0;
}

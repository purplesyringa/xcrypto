#include <lib.hpp>
#include <random>
#include <thread>

using namespace bigint;

BigInt gen_random(std::mt19937_64& mt) {
  BigInt num;
  num.data.increase_size_zerofill(1 << 16);
  for (size_t i = 0; i < num.data.size(); i++) {
    num.data[i] = mt();
  }
  return num;
}

int main() {
  std::vector<std::jthread> threads;

  for (int i = 0; i < 8; i++) {
    threads.emplace_back([i]() {
      std::mt19937_64 mt(i);

      for(;;) {
        BigInt a = gen_random(mt);
        BigInt b = gen_random(mt);

        BigInt result = a * b;

        if ((a % 179) * (b % 179) % 179 == result % 179) {
          std::cerr << ".";
        } else {
          std::cerr << "Wrong answer" << std::endl;
          std::exit(1);
        }
      }
    });
  }

  for (auto& thread: threads) {
    thread.join();
  }

  return 0;
}

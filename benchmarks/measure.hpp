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
  for (size_t base = 6; base <= 40000000; base *= 2) {
    for (size_t bits = 10 * base; bits < 20 * base; bits += base) {
      f(bits);
    }
  }
}

#include <chrono>
#include <iostream>

#include "NN/bench.hpp"

namespace bench {

UsrTime::UsrTime() : beg_(clock_::now()) {}

void UsrTime::reset() {
  beg_ = clock_::now();
}

auto UsrTime::elapsed() const -> double {
  return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
}

}  // namespace bench

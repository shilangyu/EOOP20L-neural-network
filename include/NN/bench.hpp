#include <chrono>
#include <iostream>

namespace bench {
class UsrTime {
 public:
  UsrTime();
  void reset();
  double elapsed() const;

 private:
  typedef std::chrono::high_resolution_clock clock_;
  typedef std::chrono::duration<double, std::ratio<1>> second_;
  std::chrono::time_point<clock_> beg_;
};
}  // namespace bench

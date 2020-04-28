#include <cassert>
#include <iostream>

using namespace std;

#include "./config.hpp"
#include "./functions.hpp"
// #include "./matrix.hpp"
// #include "./neural_network.hpp"

int main() {
  config_tests::init();
  functions_tests::init();
  // matrix_tests::init();
  // neural_network_tests::init();

  return 0;
}

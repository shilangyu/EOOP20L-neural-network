#include <cassert>
#include <iostream>

using namespace std;

// including all headers is a test for their correctness
#include "./config.hpp"
#include "./functions.hpp"
#include "./matrix.hpp"
#include "NN/neural_network.hpp"

int main() {
  config_tests::init();
  functions_tests::init();
  matrix_tests::init();

  return 0;
}

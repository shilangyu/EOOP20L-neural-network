#include <cassert>
#include <iostream>

using namespace std;

// including all headers is a test for their correctness
#include "./config.hpp"
#include "./functions.hpp"
#include "NN/matrix.hpp"
#include "NN/neural_network.hpp"
#include "NN/serialize.hpp"

int main() {
  config_tests::init();
  functions_tests::init();

  return 0;
}

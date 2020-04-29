#include <cassert>
#include <iostream>

#include "NN/config.hpp"
#include "NN/functions.hpp"
#include "NN/matrix.hpp"
#include "NN/neural_network.hpp"
#include "NN/serialize.hpp"

// @TODO sprinkle std::views/std::ranges/iterators or generators
// @TODO add comments
// @TODO optimize: Matrix is initialized too often

int main() {
  Config config(2, 2, 3, 4, 0.1);
  NNFunctions funcs(NNFunctions::Activation::sigmoid,
                    NNFunctions::LastLayer::softmax,
                    NNFunctions::Cost::mean_square);
  NeuralNetwork nn(config, funcs);

  std::vector<Matrix> inputs;
  std::vector<Matrix> expected;

  // false false -> false
  {
    Matrix i(2, 1);
    i[0][0] = 0.0;
    i[1][0] = 0.0;
    inputs.push_back(i);

    Matrix e(2, 1);
    e[0][0] = 1.0;
    e[1][0] = 0.0;
    expected.push_back(e);
  }

  // false true -> true
  {
    Matrix i(2, 1);
    i[0][0] = 0.0;
    i[1][0] = 1.0;
    inputs.push_back(i);

    Matrix e(2, 1);
    e[0][0] = 0.0;
    e[1][0] = 1.0;
    expected.push_back(e);
  }

  // true false -> true
  {
    Matrix i(2, 1);
    i[0][0] = 2.0;
    i[1][0] = 0.0;
    inputs.push_back(i);

    Matrix e(2, 1);
    e[0][0] = 0.0;
    e[1][0] = 1.0;
    expected.push_back(e);
  }

  // true true -> false
  {
    Matrix i(2, 1);
    i[0][0] = 1.0;
    i[1][0] = 1.0;
    inputs.push_back(i);

    Matrix e(2, 1);
    e[0][0] = 1.0;
    e[1][0] = 0.0;
    expected.push_back(e);
  }

  nn.train(inputs, expected, 10000);

  std::cout << "0 ^ 0 = " << nn.classify(inputs[0]) << std::endl;
  std::cout << "0 ^ 1 = " << nn.classify(inputs[1]) << std::endl;
  std::cout << "1 ^ 0 = " << nn.classify(inputs[2]) << std::endl;
  std::cout << "1 ^ 1 = " << nn.classify(inputs[3]) << std::endl;

  return 0;
}

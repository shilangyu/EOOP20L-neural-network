#include <cassert>
#include <iostream>
#include <vector>

#include "NN/config.hpp"
#include "NN/functions.hpp"
#include "NN/matrix.hpp"
#include "NN/neural_network.hpp"

namespace neural_network_tests {

namespace {
auto get() -> NeuralNetwork {
  Config c(2, 2, 2, 1, 0.5);
  NNFunctions f(NNFunctions::Activation::sigmoid,
                NNFunctions::LastLayer::softmax,
                NNFunctions::Cost::mean_square);
  NeuralNetwork nn(c, f);

  return nn;
}
}  // namespace

auto constructor() -> void {
  get();
}

auto classify() -> void {
  NeuralNetwork nn = get();
  Matrix inputs(2, 1);
  inputs.randomize();

  nn.classify(inputs);
}

auto train() -> void {
  NeuralNetwork nn = get();

  std::vector<Matrix> inputs;
  for (int i = 0; i < 5; i++) {
    Matrix m(2, 1);
    m.randomize();
    inputs.push_back(m);
  }

  std::vector<Matrix> expected;
  for (int i = 0; i < 5; i++) {
    Matrix m(2, 1);
    m.randomize();
    expected.push_back(m);
  }

  nn.train(inputs, expected, 100);
}

auto test() -> void {
  NeuralNetwork nn = get();

  std::vector<Matrix> inputs;
  for (int i = 0; i < 5; i++) {
    Matrix m(2, 1);
    m.randomize();
    inputs.push_back(m);
  }

  std::vector<unsigned int> expected;
  for (int i = 0; i < 5; i++) {
    expected.push_back(i);
  }

  nn.test(inputs, expected, 100);
}

auto serialize() -> void {
  auto nn = get();
  std::string serialized = nn.serialize();

  assert(false && "test not created");
  assert(serialized ==
         ("inputs=2;outputs=2;layers=2;hidden_neurons=1;learning_rate=0.5\n"
          "Activation=0;LastLayer=0;Cost=0\n"
          ""));
  /* Matrix input_w_;
std::vector<Matrix> hidden_w_;
Matrix output_w_;

/// biases of the neurons
std::vector<Matrix> hidden_b_;
Matrix output_b_; */
}

auto deserialize() -> void {
  assert(false && "test not created");
}

auto init() -> void {
  std::cout << "[neural network]" << std::endl;

  std::cout << "\t[constructor]";
  constructor();
  std::cout << "\r✅" << std::endl;

  std::cout << "\t[classify]";
  classify();
  std::cout << "\r✅" << std::endl;

  std::cout << "\t[train]";
  train();
  std::cout << "\r✅" << std::endl;

  std::cout << "\t[test]";
  test();
  std::cout << "\r✅" << std::endl;

  std::cout << "\t[serialize]";
  serialize();
  std::cout << "\r✅" << std::endl;

  std::cout << "\t[deserialize]";
  deserialize();
  std::cout << "\r✅" << std::endl;
}

}  // namespace neural_network_tests

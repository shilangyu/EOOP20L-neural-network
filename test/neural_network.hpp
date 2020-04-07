#include <cassert>
#include <iostream>
#include <vector>

using namespace std;

#include "NN/config.hpp"
#include "NN/functions.hpp"
#include "NN/matrix.hpp"
#include "NN/neural_network.hpp"

namespace neural_network_tests {

namespace {
auto get() -> NeuralNetwork {
  Config c(3, 2, 1, 4, 0.5);
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

auto guess() -> void {
  NeuralNetwork nn = get();
  Matrix inputs(3, 1);
  inputs.randomize();

  nn.classify(inputs);
}

auto train() -> void {
  NeuralNetwork nn = get();

  vector<Matrix> inputs;
  for (int i = 0; i < 5; i++) {
    Matrix m(3, 1);
    m.randomize();
    inputs.push_back(m);
  }

  vector<Matrix> expected;
  for (int i = 0; i < 5; i++) {
    Matrix m(2, 1);
    m.randomize();
    expected.push_back(m);
  }

  nn.train(inputs, expected, 100);
}

auto serialize() -> void {
  auto nn = get();
  string serialized = nn.serialize();

  assert(false && "test not created");
  assert(serialized ==
         ("inputs=3;outputs=2;layers=1;hidden_neurons=4;learning_rate=0.5\n"
          "Activation=0;LastLayer=0;Cost=0\n"
          ""));
}

auto deserialize() -> void {
  assert(false && "test not created");
}

auto init() -> void {
  cout << "[neural network]" << endl;

  cout << "\t[constructor]";
  constructor();
  cout << " ✅" << endl;

  cout << "\t[guess]";
  guess();
  cout << " ✅" << endl;

  cout << "\t[train]";
  train();
  cout << " ✅" << endl;

  cout << "\t[serialize]";
  serialize();
  cout << " ✅" << endl;

  cout << "\t[deserialize]";
  deserialize();
  cout << " ✅" << endl;
}

}  // namespace neural_network_tests

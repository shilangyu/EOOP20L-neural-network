#include <string>
#include <vector>

#include "NN/config.hpp"
#include "NN/functions.hpp"
#include "NN/matrix.hpp"
#include "NN/serialize.hpp"

using namespace std;

#pragma once

class NeuralNetwork : public Serializer<Matrix> {
 public:
  /// constructor takes the previously defined configuration
  NeuralNetwork(Config config, NNFunctions funcs);

  /// performs classification computation
  /// it is not meant for regression problems
  auto classify(const Matrix& inputs) const -> unsigned int;

  /// trains the network `n` amount of times using online training
  /// inputs and expected have to me linearly aligned: first element of inputs
  /// have to correspond to first element from expected and so on
  auto train(const vector<Matrix>& inputs,
             const vector<Matrix>& expected,
             unsigned int n) -> void;

  /// overriding the virtual methods of Serializer
  auto serialize() const -> string override;
  static auto deserialize(const string& str) -> NeuralNetwork;

 private:
  /// weights of the connections
  Matrix input_w_;
  vector<Matrix> hidden_w_;
  Matrix output_w_;

  /// biases of the neurons
  vector<Matrix> hidden_b_;
  Matrix output_b_;

  /// functions
  NNFunctions funcs_;

  /// config
  Config config_;

  /// sends inputs through the whole network and returns the output layer
  auto feedforward_(const Matrix& inputs) const -> Matrix;

  /// backpropagates the expected output from some input, adjusts the weights,
  /// then returns the cost of the network
  auto backpropagate_(const Matrix& inputs, const Matrix& expected) -> double;
};

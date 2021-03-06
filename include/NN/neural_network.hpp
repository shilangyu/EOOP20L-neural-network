#include <string>
#include <vector>

#include "NN/config.hpp"
#include "NN/functions.hpp"
#include "NN/matrix.hpp"
#include "NN/serialize.hpp"

#pragma once

class NeuralNetwork : public Serializer<NeuralNetwork> {
 public:
  /// constructor takes the previously defined configuration
  NeuralNetwork(Config config, NNFunctions funcs);

  /// performs classification computation
  /// it is not meant for regression problems
  auto classify(const Matrix& inputs) const -> unsigned int;

  /// trains the network through `n` epochs using online training (SGD)
  /// inputs and expected have to me linearly aligned: first element of inputs
  /// have to correspond to first element from expected and so on
  auto train(const std::vector<Matrix>& inputs,
             const std::vector<Matrix>& expected,
             unsigned int epochs) -> void;

  /// tests the network by running all provided samples and returns the
  /// percentage of times the classification was correct
  /// inputs and expected have to me linearly aligned: first element of inputs
  /// have to correspond to first element from expected and so on
  auto test(const std::vector<Matrix>& inputs,
            const std::vector<unsigned int>& expected) const -> double;

  /// overriding the virtual methods of Serializer
  auto serialize() const -> std::string override;
  static auto deserialize(const std::string& str) -> NeuralNetwork;

 private:
  /// weights of the connections
  Matrix input_w_;
  std::vector<Matrix> hidden_w_;
  Matrix output_w_;

  /// biases of the neurons
  std::vector<Matrix> hidden_b_;
  Matrix output_b_;

  /// functions
  NNFunctions funcs_;

  /// config
  Config config_;

  /// sends inputs through the whole network and returns values of all nodes
  auto feedforward_(const Matrix& inputs) const -> std::vector<Matrix>;

  /// backpropagates the expected output from some input, adjusts the weights,
  /// then returns the cost of the network
  auto backpropagate_(const Matrix& inputs, const Matrix& expected) -> double;
};

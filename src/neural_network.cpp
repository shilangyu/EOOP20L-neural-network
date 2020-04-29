#include <string>
#include <vector>

#include "NN/config.hpp"
#include "NN/functions.hpp"
#include "NN/matrix.hpp"
#include "NN/neural_network.hpp"

using namespace std;

NeuralNetwork::NeuralNetwork(Config config, NNFunctions funcs)
    : input_w_(config.hidden_neurons, config.inputs),
      output_w_(config.outputs, config.hidden_neurons),
      output_b_(config.outputs, 1),
      funcs_(funcs),
      config_(config) {
  input_w_.randomize();
  for (size_t i = 0; i < config.layers - 1; i++) {
    hidden_w_.emplace_back(config.hidden_neurons, config.hidden_neurons);
    hidden_w_.back().randomize();
  }
  output_w_.randomize();

  for (size_t i = 0; i < config.layers; i++) {
    hidden_b_.emplace_back(config.hidden_neurons, 1);
    hidden_b_.back().randomize();
  }
  output_b_.randomize();
}

auto NeuralNetwork::feedforward_(const Matrix& inputs) const -> Matrix {
  // @TODO allow for no hidden layers?
  Matrix h0 = input_w_ * inputs + hidden_b_[0];
  for (size_t i = 0; i < config_.hidden_neurons; i++) {
    h0[i][0] = funcs_.activation(h0[i][0]);
  }

  vector<Matrix> hiddens;
  for (size_t i = 0; i < config_.layers - 1; i++) {
    Matrix curr =
        hidden_w_[i] * (i == 0 ? h0 : hiddens[i - 1]) + hidden_b_[i + 1];
    for (size_t i = 0; i < config_.hidden_neurons; i++) {
      curr[i][0] = funcs_.activation(curr[i][0]);
    }
    hiddens.push_back(curr);
  }

  Matrix output = output_w_ * hiddens.back() + output_b_;
  vector<double> outs;
  for (size_t i = 0; i < config_.outputs; i++) {
    outs.push_back(output[i][0]);
  }
  outs = funcs_.last_layer(outs);
  for (size_t i = 0; i < config_.outputs; i++) {
    output[i][0] = outs[i];
  }

  return output;
}

auto NeuralNetwork::classify(const Matrix& inputs) const -> unsigned int {
  Matrix output = feedforward_(inputs);

  unsigned int best = 0;
  for (size_t i = 0; i < output.rows; i++) {
    if (output[i][0] > output[best][0]) {
      best = i;
    }
  }

  return best;
}

// @TODO
auto NeuralNetwork::train(const vector<Matrix>& inputs,
                          const vector<Matrix>& expected,
                          unsigned int n) -> void {
  inputs.capacity();
  expected.capacity();
  to_string(n);
}

// @TODO
auto NeuralNetwork::serialize() const -> string {
  return "";
}

// @TODO
auto NeuralNetwork::deserialize(const string& str) -> NeuralNetwork {
  str.length();
  Config config(2, 3, 2, 4, 1.0);
  NNFunctions funcs(NNFunctions::Activation::sigmoid,
                    NNFunctions::LastLayer::softmax,
                    NNFunctions::Cost::mean_square);
  return {config, funcs};
}

// @TODO
auto NeuralNetwork::backpropagate_(const Matrix& inputs, const Matrix& expected)
    -> double {
  inputs.serialize();
  expected.serialize();
  return 1.0;
}

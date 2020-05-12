#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "NN/config.hpp"
#include "NN/functions.hpp"
#include "NN/matrix.hpp"
#include "NN/neural_network.hpp"

NeuralNetwork::NeuralNetwork(Config config, NNFunctions funcs)
    : input_w_(config.hidden_neurons, config.inputs),
      output_w_(config.outputs, config.hidden_neurons),
      output_b_(config.outputs, 1),
      funcs_(funcs),
      config_(config) {
  // randomize weights
  input_w_.randomize();
  for (size_t i = 0; i < config.layers - 1; i++) {
    hidden_w_.emplace_back(config.hidden_neurons, config.hidden_neurons);
    hidden_w_.back().randomize();
  }
  output_w_.randomize();

  // randomize biases
  for (size_t i = 0; i < config.layers; i++) {
    hidden_b_.emplace_back(config.hidden_neurons, 1);
    hidden_b_.back().randomize();
  }
  output_b_.randomize();
}

auto NeuralNetwork::feedforward_(const Matrix& inputs) const
    -> std::vector<Matrix> {
  auto dot = Matrix::dot;

  std::vector<Matrix> nodes;

  // input -> first hidden layer
  nodes.push_back(dot(input_w_, inputs) + hidden_b_[0]);
  for (size_t i = 0; i < config_.hidden_neurons; i++) {
    nodes[0][i][0] = funcs_.activation(nodes[0][i][0]);
  }

  // hidden layer ->> hidden layer
  for (size_t i = 0; i < config_.layers - 1; i++) {
    Matrix curr = dot(hidden_w_[i], nodes[i]) + hidden_b_[i + 1];
    for (size_t j = 0; j < config_.hidden_neurons; j++) {
      curr[j][0] = funcs_.activation(curr[j][0]);
    }
    nodes.push_back(curr);
  }

  // last hidden layer -> output layer
  Matrix outs = dot(output_w_, nodes.back()) + output_b_;
  nodes.push_back(funcs_.last_layer(outs));

  return nodes;
}

auto NeuralNetwork::classify(const Matrix& inputs) const -> unsigned int {
  // get last nodes from feedforwading and transpose so that it's linear
  std::vector<double> output = feedforward_(inputs).back().transpose()[0];

  unsigned int best = 0;
  for (size_t i = 0; i < output.size(); i++) {
    if (output[i] > output[best]) {
      best = i;
    }
  }

  return best;
}

auto NeuralNetwork::backpropagate_(const Matrix& inputs, const Matrix& expected)
    -> double {
  auto dot = Matrix::dot;

  const std::vector<Matrix> nodes = feedforward_(inputs);
  // error is the difference of expected and actual
  // gradient is the direction of change
  // delta is the exact change that has to be applied
  std::vector<Matrix> errors, gradients, deltas;
  const double cost = funcs_.cost(expected, nodes.back());

  // last hidden layer <- output
  errors.push_back(expected - nodes.back());
  gradients.push_back(funcs_.d_last_layer(nodes.back()) * errors[0] *
                      config_.learning_rate);
  deltas.push_back(dot(gradients[0], nodes.end()[-2].transpose()));

  // hidden layer <<- hidden layer
  for (size_t i = 0; i < config_.layers - 1; i++) {
    // use output weights if first iter
    errors.push_back(dot((i == 0 ? output_w_ : hidden_w_.end()[-i]).transpose(),
                         errors.back()));
    Matrix curr = nodes.end()[-i - 2];
    for (size_t j = 0; j < curr.rows; j++) {
      curr[j][0] = funcs_.d_activation(curr[j][0]);
    }
    gradients.push_back(curr * errors.back() * config_.learning_rate);
    deltas.push_back(dot(gradients.back(), nodes.end()[-i - 3].transpose()));
  }

  // input <- first hidden layer
  // if there are no hidden weights (= there is one layer) use output weights
  errors.push_back(
      dot((config_.layers == 1 ? output_w_ : hidden_w_[0]).transpose(),
          errors.back()));
  Matrix curr = nodes[0];
  for (size_t i = 0; i < curr.rows; i++) {
    curr[i][0] = funcs_.d_activation(curr[i][0]);
  }
  gradients.push_back(curr * errors.back() * config_.learning_rate);
  deltas.push_back(dot(gradients.back(), inputs.transpose()));

  // adjust weights and biases
  output_w_ += deltas[0];
  output_b_ += gradients[0];

  for (size_t i = 0; i < config_.layers - 1; i++) {
    hidden_w_[i] += deltas[config_.layers - 1 - i];
    hidden_b_[i + 1] += gradients[config_.layers - 1 - i];
  }

  input_w_ += deltas.back();
  hidden_b_[0] += gradients.back();

  return cost;
}

auto NeuralNetwork::train(const std::vector<Matrix>& inputs,
                          const std::vector<Matrix>& expected,
                          unsigned int epochs) -> void {
  size_t n = std::min(inputs.size(), expected.size());
  std::vector<size_t> order;

  for (size_t i = 0; i < n; i++) {
    order.push_back(i);
  }

  for (size_t i = 0; i < epochs; i++) {
    std::random_shuffle(order.begin(), order.end());
    for (size_t j = 0; j < n; j++) {
      backpropagate_(inputs[order[j]], expected[order[j]]);
    }
  }
}

auto NeuralNetwork::test(const std::vector<Matrix>& inputs,
                         const std::vector<unsigned int>& expected) const
    -> double {
  size_t n = std::min(inputs.size(), expected.size());
  unsigned int n_threads = std::thread::hardware_concurrency();
  size_t samples_per_thread = std::max(n / n_threads, static_cast<size_t>(1));

  std::atomic<uint32_t> goods;
  std::vector<std::thread> handles;

  for (size_t offset = 0; offset < n; offset += samples_per_thread) {
    // if samples_per_thread got floored i need to make sure I dont overflow `n`
    size_t up_to = std::min(offset + samples_per_thread, n);

    handles.emplace_back(
        // this is safe, from and to do not overlap in different threads
        // all reads to `inputs` and `expected` will never request the same
        // memory
        [&](size_t from, size_t to) {
          for (size_t i = from; i < to; i++) {
            if (classify(inputs[i]) == expected[i]) {
              goods += 1;
            }
          }
        },
        offset, up_to);
  }

  for (auto& handle : handles) {
    handle.join();
  }

  return goods / static_cast<double>(n);
}

auto NeuralNetwork::serialize() const -> std::string {
  std::string ser;

  ser += config_.serialize() + "\n";
  ser += funcs_.serialize() + "\n";

  ser += input_w_.serialize() + "\n";
  for (size_t i = 0; i < hidden_w_.size(); i++) {
    ser += hidden_w_[i].serialize();
    if (i != hidden_w_.size() - 1) {
      ser += '|';
    }
  }
  ser += '\n';
  ser += output_w_.serialize() + "\n";

  for (size_t i = 0; i < hidden_b_.size(); i++) {
    ser += hidden_b_[i].serialize();
    if (i != hidden_b_.size() - 1) {
      ser += '|';
    }
  }
  ser += '\n';
  ser += output_b_.serialize();

  return ser;
}

auto NeuralNetwork::deserialize(const std::string& str) -> NeuralNetwork {
  auto split = [](const std::string& str, const char& delim) {
    std::string buffer;
    std::vector<std::string> chunks;

    for (auto c : str) {
      if (c != delim) {
        buffer += c;
      } else if (c == delim && buffer != "") {
        chunks.push_back(buffer);
        buffer = "";
      }
    }
    if (buffer != "") {
      chunks.push_back(buffer);
    }

    return chunks;
  };

  auto chunks = split(str, '\n');
  auto config = Config::deserialize(chunks[0]);
  auto funcs = NNFunctions::deserialize(chunks[1]);
  NeuralNetwork nn(config, funcs);

  auto apply = [](Matrix& to, Matrix other) {
    for (size_t i = 0; i < to.rows; i++) {
      for (size_t j = 0; j < to.columns; j++) {
        to[i][j] = other[i][j];
      }
    }
  };

  apply(nn.input_w_, Matrix::deserialize(chunks[2]));
  auto layers = split(chunks[3], '|');
  for (size_t i = 0; i < layers.size(); i++) {
    apply(nn.hidden_w_[i], Matrix::deserialize(layers[i]));
  }
  apply(nn.output_w_, Matrix::deserialize(chunks[4]));

  auto biases = split(chunks[5], '|');
  for (size_t i = 0; i < biases.size(); i++) {
    apply(nn.hidden_b_[i], Matrix::deserialize(biases[i]));
  }
  apply(nn.output_b_, Matrix::deserialize(chunks[6]));

  return nn;
}

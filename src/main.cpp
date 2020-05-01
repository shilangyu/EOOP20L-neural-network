#include <cassert>
#include <fstream>
#include <iostream>

#include "NN/config.hpp"
#include "NN/functions.hpp"
#include "NN/matrix.hpp"
#include "NN/neural_network.hpp"
#include "NN/serialize.hpp"

// @TODO sprinkle std::views/std::ranges/iterators or generators
// @TODO add comments
// @TODO optimize: Matrix is initialized too often

namespace mnist {
const size_t RESOLUTION = 28 * 28;

struct Digit {
  uint8_t label;
  std::array<uint8_t, RESOLUTION> pixels;
};

auto load(std::string path) -> std::vector<Digit> {
  std::ifstream f(path);

  std::vector<Digit> digits;

  std::string buff;

  while (std::getline(f, buff, '\n').good()) {
    uint8_t label = static_cast<uint8_t>(buff[0] - '0');
    std::array<uint8_t, RESOLUTION> pixels;

    std::string inner_buff;
    size_t index = 0;
    for (size_t i = 2; i < buff.size(); i++) {
      if (buff[i] == ',') {
        pixels[index++] = static_cast<uint8_t>(std::stoi(inner_buff));
        inner_buff.clear();
      } else {
        inner_buff.push_back(buff[i]);
      }
    }

    digits.emplace_back(label, pixels);
  }

  return digits;
}

namespace {
auto map_to_nn_inputs(const std::vector<Digit>& digits) -> std::vector<Matrix> {
  std::vector<Matrix> inputs;

  for (auto& d : digits) {
    Matrix input(RESOLUTION, 1);
    for (size_t i = 0; i < d.pixels.size(); i++) {
      input[i][0] = d.pixels[i] / 255.0;
    }
    inputs.push_back(input);
  }

  return inputs;
}
}  // namespace

auto map_to_nn_train(const std::vector<Digit>& digits)
    -> std::tuple<std::vector<Matrix>, std::vector<Matrix>> {
  std::vector<Matrix> inputs = map_to_nn_inputs(digits), expected;

  for (auto& d : digits) {
    Matrix label(10, 1);
    label[d.label][0] = 1.0;
    expected.push_back(label);
  }

  return std::tuple(inputs, expected);
}

auto map_to_nn_test(const std::vector<Digit>& digits)
    -> std::tuple<std::vector<Matrix>, std::vector<unsigned int>> {
  std::vector<Matrix> inputs = map_to_nn_inputs(digits);
  std::vector<unsigned int> expected;

  for (auto& d : digits) {
    expected.push_back(static_cast<unsigned int>(d.label));
  }

  return std::tuple(inputs, expected);
}
}  // namespace mnist

int main() {
  Config config(28 * 28, 10, 3, 20, 0.01);
  NNFunctions funcs(NNFunctions::Activation::sigmoid,
                    NNFunctions::LastLayer::softmax,
                    NNFunctions::Cost::mean_square);
  NeuralNetwork nn(config, funcs);

  {
    std::cout << "Loading training data..." << std::endl;
    const auto& [inputs, expected] =
        mnist::map_to_nn_train(mnist::load("mnist_train.csv"));
    std::cout << "Training..." << std::endl;
    nn.train(inputs, expected, 60000);
  }

  {
    std::cout << "Loading testing data..." << std::endl;
    const auto& [inputs, expected] =
        mnist::map_to_nn_test(mnist::load("mnist_test.csv"));
    std::cout << "Testing..." << std::endl;
    double accuracy = nn.test(inputs, expected, 10000);
    std::cout << "Final accuracy: " << accuracy * 100 << "%" << std::endl;
  }

  return 0;
}

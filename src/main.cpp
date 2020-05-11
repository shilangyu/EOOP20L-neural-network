#include <cassert>
#include <fstream>
#include <iostream>

#include "NN/config.hpp"
#include "NN/functions.hpp"
#include "NN/matrix.hpp"
#include "NN/neural_network.hpp"
#include "NN/serialize.hpp"

namespace mnist {
const size_t RESOLUTION = 28 * 28;

struct Digit {
  uint8_t label;
  std::array<uint8_t, RESOLUTION> pixels;
  Digit(uint8_t label, std::array<uint8_t, RESOLUTION> pixels)
      : label(label), pixels(pixels) {}
};

/// loads a csv from a given path and parses the content into Digits
/// it doesn't verify if the file is correct
auto load(std::string path) -> std::vector<Digit> {
  std::ifstream f(path);

  std::vector<Digit> digits;

  std::string buff;

  while (std::getline(f, buff, '\n').good()) {
    // label is always the first char in a line
    uint8_t label = static_cast<uint8_t>(buff[0] - '0');
    std::array<uint8_t, RESOLUTION> pixels;

    uint8_t curr = 0;
    size_t index = 0;
    // starting from 2 to because 0 is the label and 1 is a comma
    for (size_t i = 2; i < buff.size(); i++) {
      if (buff[i] == ',') {
        pixels[index++] = curr;
        curr = 0;
      } else {
        // example why this works:
        // curr = 0
        // read 2 => curr *= 10 (curr = 0) => curr += 2 (curr = 2)
        // read 1 => curr *= 10 (curr = 20) => curr += 1 (curr = 21)
        // read 9 => curr *= 10 (curr = 210) => curr += 9 (curr = 219)
        curr *= 10;
        curr += static_cast<uint8_t>(buff[i] - '0');
      }
    }

    digits.emplace_back(label, pixels);
  }

  return digits;
}

namespace {
/// takes digits and turns them into inputs that can be fed to the NN
auto map_to_nn_inputs(const std::vector<Digit>& digits) -> std::vector<Matrix> {
  std::vector<Matrix> inputs(digits.size(), Matrix(RESOLUTION, 1));

  for (size_t i = 0; i < inputs.size(); i++) {
    for (size_t j = 0; j < digits[i].pixels.size(); j++) {
      // normalizing
      inputs[i][j][0] = digits[i].pixels[j] / 255.0;
    }
  }

  return inputs;
}
}  // namespace

/// maps digits to a tuple of inputs and expected outputs
/// that can be fed to the NN
auto map_to_nn_train(const std::vector<Digit>& digits)
    -> std::tuple<std::vector<Matrix>, std::vector<Matrix>> {
  std::vector<Matrix> inputs = map_to_nn_inputs(digits),
                      expected(digits.size(), Matrix(10, 1));

  for (size_t i = 0; i < expected.size(); i++) {
    expected[i][digits[i].label][0] = 1.0;
  }

  return std::tuple(inputs, expected);
}

/// maps digits to a tuple of inputs and expected classifications
/// that can be fed to the NN
auto map_to_nn_test(const std::vector<Digit>& digits)
    -> std::tuple<std::vector<Matrix>, std::vector<unsigned int>> {
  std::vector<Matrix> inputs = map_to_nn_inputs(digits);
  std::vector<unsigned int> expected;
  expected.reserve(digits.size());

  for (auto& d : digits) {
    expected.push_back(static_cast<unsigned int>(d.label));
  }

  return std::tuple(inputs, expected);
}
}  // namespace mnist

int main() {
  Config config(mnist::RESOLUTION, 10, 2, 16, 0.01);
  NNFunctions funcs(NNFunctions::Activation::sigmoid,
                    NNFunctions::LastLayer::softmax,
                    NNFunctions::Cost::mean_square);
  NeuralNetwork nn(config, funcs);

  {
    std::cout << "Loading training data..." << std::endl;
    const auto& [inputs, expected] =
        mnist::map_to_nn_train(mnist::load("mnist_train.csv"));
    std::cout << "Training..." << std::endl;
    nn.train(inputs, expected, 100000);
    nn.to_file("brain.txt");
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

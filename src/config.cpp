#include <cstdio>
#include <string>

#include "NN/config.hpp"

Config::Config(const unsigned int inputs,
               const unsigned int outputs,
               const unsigned int layers,
               const unsigned int hidden_neurons,
               const double learning_rate)
    : inputs(inputs),
      outputs(outputs),
      layers(layers),
      hidden_neurons(hidden_neurons),
      learning_rate(learning_rate) {}

auto Config::serialize() const -> std::string {
  return "inputs=" + std::to_string(inputs) + ';' +
         "outputs=" + std::to_string(outputs) + ';' +
         "layers=" + std::to_string(layers) + ';' +
         "hidden_neurons=" + std::to_string(hidden_neurons) + ';' +
         "learning_rate=" + std::to_string(learning_rate);
}
auto Config::deserialize(const std::string& str) -> Config {
  unsigned int inputs, outputs, layers, hidden_neurons;
  double learning_rate;

  sscanf(str.c_str(),
         "inputs=%u;outputs=%u;layers=%u;hidden_neurons=%u;learning_rate=%lf",
         &inputs, &outputs, &layers, &hidden_neurons, &learning_rate);

  return Config{inputs, outputs, layers, hidden_neurons, learning_rate};
}

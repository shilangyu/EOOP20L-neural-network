#include <cstdio>
#include <string>

#include "NN/config.hpp"

using namespace std;

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

auto Config::serialize() const -> string {
  return "inputs=" + to_string(inputs) + ';' + "outputs=" + to_string(outputs) +
         ';' + "layers=" + to_string(layers) + ';' +
         "hidden_neurons=" + to_string(hidden_neurons) + ';' +
         "learning_rate=" + to_string(learning_rate);
}
auto Config::deserialize(const string& str) -> Config {
  unsigned int inputs, outputs, layers, hidden_neurons;
  double learning_rate;

  sscanf(str.c_str(),
         "inputs=%u;outputs=%u;layers=%u;hidden_neurons=%u;learning_rate=%lf",
         &inputs, &outputs, &layers, &hidden_neurons, &learning_rate);

  return Config{inputs, outputs, layers, hidden_neurons, learning_rate};
}

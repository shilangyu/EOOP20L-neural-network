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
  string serialized;
  serialized += "inputs=" + to_string(inputs) + ';';
  serialized += "outputs=" + to_string(outputs) + ';';
  serialized += "layers=" + to_string(layers) + ';';
  serialized += "hidden_neurons=" + to_string(hidden_neurons) + ';';
  serialized += "learning_rate=" + to_string(learning_rate);

  return serialized;
}
auto Config::deserialize(const string& str) -> Config {
  unsigned int inputs, outputs, layers, hidden_neurons;
  double learning_rate;

  sscanf(str.c_str(),
         "inputs=%u;outputs=%u;layers=%u;hidden_neurons=%u;learning_rate=%lf",
         &inputs, &outputs, &layers, &hidden_neurons, &learning_rate);

  return Config{inputs, outputs, layers, hidden_neurons, learning_rate};
}

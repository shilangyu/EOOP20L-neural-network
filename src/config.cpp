#include <cstdio>
#include <string>

#include "NN/config.hpp"
#include "NN/serialize.hpp"

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
  serialized += to_string(inputs) + ',';
  serialized += to_string(outputs) + ',';
  serialized += to_string(layers) + ',';
  serialized += to_string(hidden_neurons) + ',';
  serialized += to_string(learning_rate);

  return serialized;
}
auto Config::deserialize(const string& str) -> Config {
  unsigned int inputs, outputs, layers, hidden_neurons;
  double learning_rate;

  sscanf(str.c_str(), "%u,%u,%u,%u,%lf", &inputs, &outputs, &layers,
         &hidden_neurons, &learning_rate);

  return Config{inputs, outputs, layers, hidden_neurons, learning_rate};
}

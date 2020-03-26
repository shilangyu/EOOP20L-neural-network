#include <string>

#include "NN/serialize.hpp"

using namespace std;

#pragma once

class Config : Serializer<Config> {
 public:
  /// properties of a neural network
  /// because they are constant, there is no need encapsulating them
  const unsigned int inputs, outputs, layers, hidden_neurons;
  const double learning_rate;

  /// constructor accepting all 4 parameters
  Config(const unsigned int inputs, const unsigned int outputs,
         const unsigned int layers, const unsigned int hidden_neurons,
         const double learning_rate);

  /// overriding the virtual methods of Serializer
  static auto deserialize(string str) -> Config;
  auto serialize() -> string override;
};

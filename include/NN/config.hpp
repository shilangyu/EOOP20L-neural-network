#include <string>

#include "NN/serialize.hpp"

#pragma once

class Config : public Serializer<Config> {
 public:
  /// properties of a neural network
  /// because they are constant, there is no need encapsulating them
  const unsigned int inputs, outputs, layers, hidden_neurons;
  const double learning_rate;

  /// constructor accepting all 5 parameters
  Config(const unsigned int inputs,
         const unsigned int outputs,
         const unsigned int layers,
         const unsigned int hidden_neurons,
         const double learning_rate);

  /// overriding the virtual methods of Serializer
  auto serialize() const -> std::string override;
  static auto deserialize(const std::string& str) -> Config;
};

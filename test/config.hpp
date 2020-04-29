#include <cassert>
#include <iostream>

#include "NN/config.hpp"

namespace config_tests {

auto constructor() -> void {
  Config c(1, 2, 3, 4, 0.5);

  assert(c.inputs == 1);
  assert(c.outputs == 2);
  assert(c.layers == 3);
  assert(c.hidden_neurons == 4);
  assert(c.learning_rate == 0.5);
}

auto serialize() -> void {
  Config c(1, 2, 3, 4, 0.5);
  std::string serialized = c.serialize();

  assert(serialized ==
         "inputs=1;outputs=2;layers=3;hidden_neurons=4;learning_rate=0.500000");
}

auto deserialize() -> void {
  auto c = Config::deserialize(
      "inputs=1;outputs=2;layers=3;hidden_neurons=4;learning_rate=0.5");

  assert(c.inputs == 1);
  assert(c.outputs == 2);
  assert(c.layers == 3);
  assert(c.hidden_neurons == 4);
  assert(c.learning_rate == 0.5);
}

auto init() -> void {
  std::cout << "[config]" << std::endl;

  std::cout << "\t[constructor]";
  constructor();
  std::cout << "\r✅" << std::endl;

  std::cout << "\t[serialize]";
  serialize();
  std::cout << "\r✅" << std::endl;

  std::cout << "\t[deserialize]";
  deserialize();
  std::cout << "\r✅" << std::endl;
}

}  // namespace config_tests

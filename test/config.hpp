#include <cassert>
#include <iostream>

using namespace std;

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
  string serialized = c.serialize();

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
  cout << "[config]" << endl;

  cout << "\t[constructor]";
  constructor();
  cout << " ✅" << endl;

  cout << "\t[serialize]";
  serialize();
  cout << " ✅" << endl;

  cout << "\t[deserialize]";
  deserialize();
  cout << " ✅" << endl;
}

}  // namespace config_tests

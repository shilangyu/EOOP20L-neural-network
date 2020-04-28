#include <cassert>
#include <iostream>

using namespace std;

#include "NN/functions.hpp"

namespace functions_tests {

auto constructors() -> void {
  NNFunctions f(NNFunctions::Activation::relu, NNFunctions::LastLayer::softmax,
                NNFunctions::Cost::mean_square);

  assert(f.activation != NULL);
  assert(f.d_activation != NULL);
  assert(f.last_layer != NULL);
  assert(f.d_last_layer != NULL);
  assert(f.cost != NULL);

  NNFunctions ff(
      [](double v) { return v; }, [](double v) { return v; },
      [](vector<double> v) { return v; }, [](vector<double> v) { return v; },
      [](const vector<double>& v1, const vector<double>& v2) {
        v1.capacity();  /// done to avoid the 'unused variable' warning
        v2.capacity();  /// done to avoid the 'unused variable' warning
        return 1.0;
      });

  assert(ff.activation != NULL);
  assert(ff.d_activation != NULL);
  assert(ff.last_layer != NULL);
  assert(ff.d_last_layer != NULL);
  assert(ff.cost != NULL);
}

auto serialize() -> void {
  NNFunctions f(NNFunctions::Activation::relu, NNFunctions::LastLayer::softmax,
                NNFunctions::Cost::mean_square);

  assert(f.serialize() == "Activation=1;LastLayer=0;Cost=0");

  NNFunctions ff(
      [](double v) { return v; }, [](double v) { return v; },
      [](vector<double> v) { return v; }, [](vector<double> v) { return v; },
      [](const vector<double>& v1, const vector<double>& v2) {
        v1.capacity();  /// done to avoid the 'unused variable' warning
        v2.capacity();  /// done to avoid the 'unused variable' warning
        return 1.0;
      });

  assert(ff.serialize() == "Activation=3;LastLayer=1;Cost=1");
}

auto deserialize() -> void {
  auto f = NNFunctions::deserialize("Activation=2;LastLayer=0;Cost=0");

  assert(f.activation != NULL);
  assert(f.d_activation != NULL);
  assert(f.last_layer != NULL);
  assert(f.d_last_layer != NULL);
  assert(f.cost != NULL);
}

auto calling() -> void {
  NNFunctions ff(
      [](double v) { return v + 1.0; }, [](double v) { return v + 2.0; },
      [](vector<double> v) { return v; }, [](vector<double> v) { return v; },
      [](const vector<double>& v1, const vector<double>& v2) {
        v1.capacity();  /// done to avoid the 'unused variable' warning
        v2.capacity();  /// done to avoid the 'unused variable' warning
        return 3.2;
      });

  assert(ff.activation(1.2) == 2.2);
  assert(ff.d_activation(1.2) == 3.2);
  vector<double> v = {1.0, 2.0};
  assert(ff.last_layer(v).size() == 2);
  assert(ff.d_last_layer(v).size() == 2);
  assert(ff.cost(v, v) == 3.2);
}

auto init() -> void {
  cout << "[functions]" << endl;

  cout << "\t[constructor]";
  constructors();
  cout << " ✅" << endl;

  cout << "\t[serialize]";
  serialize();
  cout << " ✅" << endl;

  cout << "\t[deserialize]";
  deserialize();
  cout << " ✅" << endl;

  cout << "\t[calling]";
  calling();
  cout << " ✅" << endl;
}

}  // namespace functions_tests

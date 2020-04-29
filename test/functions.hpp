#include <cassert>
#include <iostream>

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
      [](Matrix m) { return m; }, [](Matrix m) { return m; },
      [](const Matrix& m1, const Matrix& m2) {
        m1.transpose();  /// done to avoid the 'unused variable' warning
        m2.transpose();  /// done to avoid the 'unused variable' warning
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
      [](Matrix m) { return m; }, [](Matrix m) { return m; },
      [](const Matrix& m1, const Matrix& m2) {
        m1.transpose();  /// done to avoid the 'unused variable' warning
        m2.transpose();  /// done to avoid the 'unused variable' warning
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
      [](Matrix m) { return m; }, [](Matrix m) { return m; },
      [](const Matrix& m1, const Matrix& m2) {
        m1.transpose();  /// done to avoid the 'unused variable' warning
        m2.transpose();  /// done to avoid the 'unused variable' warning
        return 3.2;
      });

  assert(ff.activation(1.2) == 2.2);
  assert(ff.d_activation(1.2) == 3.2);
  Matrix m(2, 1);
  assert(ff.last_layer(m).rows == 2);
  assert(ff.d_last_layer(m).rows == 2);
  assert(ff.cost(m, m) == 3.2);
}

auto init() -> void {
  std::cout << "[functions]" << std::endl;

  std::cout << "\t[constructor]";
  constructors();
  std::cout << "\r✅" << std::endl;

  std::cout << "\t[serialize]";
  serialize();
  std::cout << "\r✅" << std::endl;

  std::cout << "\t[deserialize]";
  deserialize();
  std::cout << "\r✅" << std::endl;

  std::cout << "\t[calling]";
  calling();
  std::cout << "\r✅" << std::endl;
}

}  // namespace functions_tests

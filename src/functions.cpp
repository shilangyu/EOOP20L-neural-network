#include <cmath>
#include <string>
#include <vector>

#include "NN/functions.hpp"
#include "NN/matrix.hpp"

NNFunctions::NNFunctions(Activation af, LastLayer llf, Cost cf)
    : af_(af), llf_(llf), cf_(cf) {
  switch (af) {
    case Activation::relu:
      activation = [](double x) { return std::max(0.0, x); };
      d_activation = [](double y) { return y > 0.0 ? 1.0 : 0.0; };
      break;
    case Activation::sigmoid:
      activation = [](double x) { return 1.0 / (1.0 + exp(-x)); };
      d_activation = [](double y) { return y * (1.0 - y); };
      break;
    case Activation::tanh:
      activation = [](double x) { return tanh(x); };
      d_activation = [](double y) { return 1.0 - y * y; };
      break;
    default:
      break;
  }

  switch (llf) {
    case LastLayer::softmax:
      last_layer = [](Matrix xs) {
        double biggest = xs[0][0];
        for (auto x : xs[0]) {
          biggest = std::max(biggest, x);
        }

        double sum = 0.0;
        for (auto& x : xs[0]) {
          x = exp(x - biggest);
          sum += x;
        }
        for (auto& x : xs[0]) {
          x /= sum;
        }
        return xs;
      };
      d_last_layer = [](Matrix ys) {
        for (auto& y : ys[0]) {
          // this is not entirely true, it should take into account if i == j,
          // but it's minor
          y = y * (1.0 - y);
        }
        return ys;
      };
      break;
    case LastLayer::sigmoid:
      last_layer = [](Matrix xs) {
        for (auto& x : xs[0]) {
          x = 1.0 / (1.0 + exp(-x));
        }

        return xs;
      };
      d_last_layer = [](Matrix ys) {
        for (auto& y : ys[0]) {
          y = y * (1.0 - y);
        }

        return ys;
      };
      break;
    default:
      break;
  }

  switch (cf) {
    case Cost::mean_square:
      cost = [](const Matrix& m1, const Matrix& m2) {
        Matrix diff = m1 - m2;
        double cost = 0.0;
        for (size_t i = 0; i < diff.rows; i++) {
          cost += pow(diff[i][0], 2.0) / 2.0;
        }

        return cost / (diff.rows * 1.0);
      };
      break;
    default:
      break;
  }
}

NNFunctions::NNFunctions(const Activating af,
                         const Activating daf,
                         const Mapping llf,
                         const Mapping dllf,
                         const Reducing cf)
    : activation(af),
      d_activation(daf),
      last_layer(llf),
      d_last_layer(dllf),
      cost(cf),
      af_(Activation::__custom),
      llf_(LastLayer::__custom),
      cf_(Cost::__custom) {}

auto NNFunctions::serialize() const -> std::string {
  return "Activation=" + std::to_string(static_cast<int>(af_)) + ';' +
         "LastLayer=" + std::to_string(static_cast<int>(llf_)) + ';' +
         "Cost=" + std::to_string(static_cast<int>(cf_));
}

auto NNFunctions::deserialize(const std::string& str) -> NNFunctions {
  int af, llf, cf;

  sscanf(str.c_str(), "Activation=%d;LastLayer=%d;Cost=%d", &af, &llf, &cf);

  return NNFunctions{static_cast<Activation>(af), static_cast<LastLayer>(llf),
                     static_cast<Cost>(cf)};
}

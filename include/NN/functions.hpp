#include <string>

using namespace std;

#pragma once

enum class ActivationFunction { sigmoid, relu, tanh };

enum class LastLayerFunction { softmax };

enum class CostFunction { mean_square };

class NNFunctions {
 public:
  /// type definitions of the functions
  /// a function that takes a double and decides if its active
  typedef auto (*Activating)(double&) -> double;
  /// a function that takes an array of doubles and maps it to different values
  typedef auto (*Mapping)(double&) -> double;
  /// a function that takes an array of doubles and reduces it to a single value
  typedef auto (*Reducing)(double&) -> double;

  /// collection of functions
  const Activating activation, d_activation;
  const Mapping last_layer, d_last_layer;
  const Reducing cost;

  /// constructor accepting enums describing pre-made functions
  NNFunctions(ActivationFunction af, LastLayerFunction llf, CostFunction cf);
  /// constructor accepting functions
  NNFunctions(const Activating af, const Activating daf, const Mapping llf,
              const Mapping dllf, const Reducing cf);
};

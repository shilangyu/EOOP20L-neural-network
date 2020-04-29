#include <functional>
#include <string>
#include <vector>

#include "NN/matrix.hpp"
#include "NN/serialize.hpp"

using namespace std;

#pragma once

class NNFunctions : public Serializer<NNFunctions> {
 public:
  /// enums listing available function
  /// __custom means the function was provided
  enum class Activation { sigmoid, relu, tanh, __custom };
  enum class LastLayer { softmax, __custom };
  enum class Cost { mean_square, __custom };

  /// type definitions of the functions
  /// a function that takes a double and decides if its active
  typedef function<auto(double)->double> Activating;
  /// a function that takes an matrix and maps it to different values
  typedef function<auto(Matrix)->Matrix> Mapping;
  /// a function that takes two arrays of doubles and reduces it to a single
  /// value
  typedef function<auto(const vector<double>&, const vector<double>&)->double>
      Reducing;

  /// collection of functions
  Activating activation, d_activation;
  Mapping last_layer, d_last_layer;
  Reducing cost;

  /// constructor accepting enums describing pre-made functions
  NNFunctions(Activation af, LastLayer llf, Cost cf);
  /// constructor accepting functions
  NNFunctions(const Activating af,
              const Activating daf,
              const Mapping llf,
              const Mapping dllf,
              const Reducing cf);

  /// overriding the virtual methods of Serializer
  auto serialize() const -> string override;
  static auto deserialize(const string& str) -> NNFunctions;

 private:
  /// remembering which functions were chosen, this information is needed for
  /// serialization
  Activation af_;
  LastLayer llf_;
  Cost cf_;
};

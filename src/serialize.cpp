#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "NN/serialize.hpp"

// sadly generics in cpp have a weird compilation process,
// I have to explicitely declare them
#include "NN/config.hpp"
#include "NN/functions.hpp"
#include "NN/matrix.hpp"
#include "NN/neural_network.hpp"
template class Serializer<Config>;
template class Serializer<NNFunctions>;
template class Serializer<Matrix>;
template class Serializer<NeuralNetwork>;

template <typename T>
auto Serializer<T>::from_file(const std::string& path) -> T {
  std::ifstream file(path);

  // @TODO error handling
  if (!file) {
  }

  std::stringstream ss;
  ss << file.rdbuf();
  std::string contents = ss.str();

  return T::deserialize(contents);
}

template <typename T>
auto Serializer<T>::to_file(const std::string& path) const -> void {
  std::ofstream file(path);

  // @TODO error handling
  if (!file) {
  }

  file << serialize() << std::endl;
}

//  friend auto operator<<<>(ostream& os, const T& obj) -> ostream&;

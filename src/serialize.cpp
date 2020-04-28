#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "NN/serialize.hpp"

using namespace std;

template <typename T>
auto Serializer<T>::from_file(const string& path) -> T {
  ifstream file(path);

  // @TODO error handling
  if (!file) {
  }

  stringstream ss;
  ss << file.rdbuf();
  string contents = ss.str();

  return T::deserialize(contents);
}

template <typename T>
auto Serializer<T>::to_file(const string& path) const -> void {
  ofstream file(path);

  // @TODO error handling
  if (!file) {
  }

  file << serialize() << endl;
}

//  friend auto operator<<<>(ostream& os, const T& obj) -> ostream&;

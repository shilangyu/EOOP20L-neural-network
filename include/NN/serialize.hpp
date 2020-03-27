#include <iostream>
#include <string>

using namespace std;

#pragma once

template <typename T>
class Serializer {
 protected:
  /// protected on purpose, Serializer is an abstract class
  Serializer();

 public:
  /// deserializes a file into the parent object.
  /// throws if file does not exist
  static auto from_file(const string& path) -> T;

  /// takes a path and serializes the parent into the pointed file.
  /// overwrites all content if the file already exists
  auto to_file(const string& path) const -> void;

  /// prints the serialized object
  // friend auto operator<<<>(ostream& os, const T& obj) -> ostream&;

  /// virtual methods that have to be implemented by parent classes
  /// then the serializer can work properly
  virtual auto serialize() const -> string = 0;
  /// this is impossible in c++, therefor it only serves as a documentation
  // virtual static auto deserialize(const string& str) -> T = 0;
};

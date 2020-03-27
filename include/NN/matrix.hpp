#include <string>
#include <vector>

#include "NN/serialize.hpp"

using namespace std;

#pragma once

class Matrix : Serializer<Matrix> {
 public:
  /// size of the matrix
  const unsigned int rows, columns;

  /// constructor takes the dimensions of the matrix
  Matrix(const unsigned int rows, const unsigned int columns);

  /// randomizes the matrix with a given range
  auto randomize(const double min = -1.0, const double max = 1.0) -> void;

  /// operator overloads for matrix operations
  /// in-place
  auto operator+=(const Matrix& rhs) -> Matrix&;
  auto operator+=(const double& rhs) -> Matrix&;
  auto operator-=(const Matrix& rhs) -> Matrix&;
  auto operator-=(const double& rhs) -> Matrix&;
  auto operator*=(const double& rhs) -> Matrix&;
  auto operator/=(const double& rhs) -> Matrix&;
  /// global
  friend auto operator+(const Matrix& lhs, const Matrix& rhs) -> Matrix;
  friend auto operator+(const Matrix& lhs, const double& rhs) -> Matrix;
  friend auto operator-(const Matrix& lhs, const Matrix& rhs) -> Matrix;
  friend auto operator-(const Matrix& lhs, const double& rhs) -> Matrix;
  friend auto operator*(const Matrix& lhs, const Matrix& rhs) -> Matrix;
  friend auto operator*(const Matrix& lhs, const double& rhs) -> Matrix;
  friend auto operator/(const Matrix& lhs, const double& rhs) -> Matrix;
  /// indexing
  auto operator[](size_t idx) const -> vector<double>&;

  /// transposing flips the x and y axis
  auto transpose() const -> Matrix;

  /// overriding the virtual methods of Serializer
  auto serialize() const -> string override;
  static auto deserialize(const string& str) -> Matrix;

 private:
  /// thats where the data is stored. Vector was chosen because while the
  /// size is immutable and array would seem like a more fitting choice, vector
  /// provides a much safer interface with almost no overhead
  vector<vector<double>> data_;
};

#include <string>
#include <vector>

#include "NN/serialize.hpp"

#pragma once

class Matrix : public Serializer<Matrix> {
 public:
  /// size of the matrix
  const unsigned int rows, columns;

  /// custom exception for size mismatches when performing operations
  struct SizeMismatch : public std::exception {
    std::string msg;

    SizeMismatch(const Matrix& m1,
                 const Matrix& m2,
                 const std::string op_name) {
      msg = "Cannot perform operation with mismatched sizes: m1(" +
            std::to_string(m1.rows) + ", " + std::to_string(m1.columns) + ") " +
            op_name + " m2(" + std::to_string(m2.rows) + ", " +
            std::to_string(m2.columns) + ")";
    }

    const char* what() const throw() { return msg.c_str(); }
  };

  /// constructor takes the dimensions of the matrix
  Matrix(const unsigned int rows, const unsigned int columns);

  /// randomizes the matrix with a given range
  auto randomize(const double min = -1.0, const double max = 1.0) -> void;

  /// operator overloads for matrix operations
  /// if on both sides of the operation theres a matrix then the operation is
  /// done element wise, unless it is * where a matrix multiplication is
  /// performed instead
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
  auto operator[](size_t idx) -> std::vector<double>&;

  /// transposing flips the x and y axis
  auto transpose() const -> Matrix;

  /// overriding the virtual methods of Serializer
  auto serialize() const -> std::string override;
  static auto deserialize(const std::string& str) -> Matrix;

 private:
  /// thats where the data is stored. Vector was chosen because while the
  /// size is immutable and array would seem like a more fitting choice, vector
  /// provides a much safer interface with negligible overhead
  std::vector<std::vector<double>> data_;

  /// throws SizeMismatch if rows and columns are different
  static auto ensure_same_size_(const Matrix& m1,
                                const Matrix& m2,
                                const std::string op_name) -> void;
};

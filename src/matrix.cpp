#include <random>
#include <vector>

#include "NN/matrix.hpp"

using namespace std;

Matrix::Matrix(const unsigned int rows, const unsigned int columns)
    : rows(rows), columns(columns), data_(vector(rows, vector(columns, 0.0))) {}

auto Matrix::randomize(const double min, const double max) -> void {
  static random_device rd;
  static default_random_engine engine(rd());
  uniform_real_distribution<> dis(min, max);

  for (auto& row : data_) {
    for (auto& x : row) {
      x = dis(engine);
    }
  }
}

auto Matrix::ensure_same_size_(const Matrix& m1,
                               const Matrix& m2,
                               const string op_name) -> void {
  if (m1.rows != m2.rows || m1.columns != m2.columns) {
    throw SizeMismatch(m1, m2, op_name);
  }
}

auto Matrix::operator+=(const Matrix& rhs) -> Matrix& {
  ensure_same_size_(*this, rhs, "+=");

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < columns; j++) {
      data_[i][j] += rhs.data_[i][j];
    }
  }

  return *this;
}

auto Matrix::operator+=(const double& rhs) -> Matrix& {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < columns; j++) {
      data_[i][j] += rhs;
    }
  }

  return *this;
}

auto Matrix::operator-=(const Matrix& rhs) -> Matrix& {
  ensure_same_size_(*this, rhs, "-=");

  *this += rhs * -1.0;

  return *this;
}

auto Matrix::operator-=(const double& rhs) -> Matrix& {
  *this += -rhs;

  return *this;
}

auto Matrix::operator*=(const double& rhs) -> Matrix& {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < columns; j++) {
      data_[i][j] *= rhs;
    }
  }

  return *this;
}

auto Matrix::operator/=(const double& rhs) -> Matrix& {
  *this *= 1.0 / rhs;

  return *this;
}

auto operator+(const Matrix& lhs, const Matrix& rhs) -> Matrix {
  Matrix::ensure_same_size_(lhs, rhs, "+");

  Matrix m = lhs;
  m += rhs;

  return m;
}

auto operator-(const Matrix& lhs, const Matrix& rhs) -> Matrix {
  Matrix::ensure_same_size_(lhs, rhs, "-");

  return lhs + rhs * -1.0;
}

auto operator+(const Matrix& lhs, const double& rhs) -> Matrix {
  Matrix m = lhs;
  m += rhs;
  return m;
}

auto operator-(const Matrix& lhs, const double& rhs) -> Matrix {
  return lhs + (-rhs);
}

auto operator*(const Matrix& lhs, const Matrix& rhs) -> Matrix {
  if (lhs.columns != rhs.rows) {
    throw Matrix::SizeMismatch(lhs, rhs, "*");
  }

  Matrix m(lhs.rows, rhs.columns);

  for (size_t k = 0; k < m.columns; k++) {
    for (size_t i = 0; i < m.rows; i++) {
      for (size_t j = 0; j < lhs.columns; j++) {
        m[i][k] += (lhs.data_[i][j] * rhs.data_[j][k]);
      }
    }
  }

  return m;
}

auto operator*(const Matrix& lhs, const double& rhs) -> Matrix {
  Matrix m = lhs;
  m *= rhs;
  return m;
}

auto operator/(const Matrix& lhs, const double& rhs) -> Matrix {
  return lhs * (1.0 / rhs);
}

auto Matrix::operator[](size_t idx) -> vector<double>& {
  return data_[idx];
}

auto Matrix::transpose() const -> Matrix {
  Matrix transposed(columns, rows);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < columns; j++) {
      transposed[j][i] = data_[i][j];
    }
  }

  return transposed;
}

auto Matrix::serialize() const -> string {
  string ser;

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < columns; j++) {
      ser += to_string(data_[i][j]);
      if (j != columns - 1) {
        ser += ',';
      }
    }
    if (i != rows - 1) {
      ser += ';';
    }
  }

  return ser;
}

auto Matrix::deserialize(const string& str) -> Matrix {
  auto split = [](const string& str, const char& delim) {
    string buffer;
    vector<string> chunks;

    for (auto c : str) {
      if (c != delim) {
        buffer += c;
      } else if (c == delim && buffer != "") {
        chunks.push_back(buffer);
        buffer = "";
      }
    }
    if (buffer != "") {
      chunks.push_back(buffer);
    }

    return chunks;
  };

  vector<vector<double>> values;
  for (auto& row : split(str, ';')) {
    values.push_back(vector<double>());
    for (auto& num : split(row, ',')) {
      values.back().push_back(stod(num));
    }
  }

  Matrix m(values.size(), values[0].size());

  for (size_t i = 0; i < m.rows; i++) {
    m[i] = values[i];
  }

  return m;
}

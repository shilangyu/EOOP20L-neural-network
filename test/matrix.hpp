#include <cassert>
#include <iostream>

using namespace std;

#include "NN/matrix.hpp"

namespace matrix_tests {

auto constructor() -> void {
  Matrix m(1, 2);

  assert(m.rows == 1);
  assert(m.columns == 2);
}

auto randomize() -> void {
  Matrix m(100, 200);
  m.randomize(10.0, 11.0);

  for (unsigned int i = 0; i < m.rows; i++) {
    for (auto val : m[i]) {
      assert(val < 11.0 && val > 10.0);
    }
  }
}

auto in_place_operators() -> void {
  auto get = [](int rows = 1) {
    Matrix m(rows, 2);

    m[0][0] = 1.0;
    m[0][1] = 2.0;

    return m;
  };

  {
    Matrix m1 = get();
    Matrix m2 = get();

    m1 += m2;

    assert(m1[0][0] == 2.0);
    assert(m1[0][1] == 4.0);
  }

  {
    Matrix m1 = get();
    Matrix m2 = get(2);

    try {
      m1 += m2;
      assert(false);
    } catch (...) {
    }
  }

  {
    Matrix m1 = get();

    m1 += 2.0;

    assert(m1[0][0] == 3.0);
    assert(m1[0][1] == 4.0);
  }

  {
    Matrix m1 = get();
    Matrix m2 = get();

    m1 -= m2;

    assert(m1[0][0] == 0.0);
    assert(m1[0][1] == 0.0);
  }

  {
    Matrix m1 = get();
    Matrix m2 = get(2);

    try {
      m1 -= m2;
      assert(false);
    } catch (...) {
    }
  }

  {
    Matrix m1 = get();

    m1 -= 2.0;

    assert(m1[0][0] == -1.0);
    assert(m1[0][1] == 0.0);
  }

  {
    Matrix m1 = get();

    m1 *= 2.0;

    assert(m1[0][0] == 2.0);
    assert(m1[0][1] == 4.0);
  }

  {
    Matrix m1 = get();

    m1 /= 2.0;

    assert(m1[0][0] == 0.5);
    assert(m1[0][1] == 1.0);
  }
}

auto global_operators() -> void {  // @TODO
  auto get = [](int rows = 1) {
    Matrix m(rows, 2);

    m[0][0] = 1.0;
    m[0][1] = 2.0;

    return m;
  };

  {
    Matrix m1 = get();
    Matrix m2 = get();

    Matrix m3 = m1 + m2;

    assert(m3[0][0] == 2.0);
    assert(m3[0][1] == 4.0);
  }

  {
    Matrix m1 = get();
    Matrix m2 = get(2);

    try {
      Matrix m3 = m1 + m2;
      assert(false);
    } catch (...) {
    }
  }

  {
    Matrix m1 = get();

    Matrix m3 = m1 + 2.0;

    assert(m3[0][0] == 3.0);
    assert(m3[0][1] == 4.0);
  }

  {
    Matrix m1 = get();
    Matrix m2 = get();

    Matrix m3 = m1 - m2;

    assert(m3[0][0] == 0.0);
    assert(m3[0][1] == 0.0);
  }

  {
    Matrix m1 = get();
    Matrix m2 = get(2);

    try {
      Matrix m3 = m1 - m2;
      assert(false);
    } catch (...) {
    }
  }

  {
    Matrix m1 = get();

    Matrix m3 = m1 - 2.0;

    assert(m3[0][0] == -1.0);
    assert(m3[0][1] == 0.0);
  }

  {
    Matrix m1 = get();
    Matrix m2 = get(2);
    m2[1][0] = 3.0;

    Matrix m3 = m1 * m2;

    assert(m3[0][0] == 7.0);
    assert(m3[0][1] == 2.0);
  }

  {
    Matrix m1 = get();
    Matrix m2 = get(3);

    try {
      Matrix m3 = m1 * m2;
      assert(false);
    } catch (...) {
    }
  }

  {
    Matrix m1 = get();

    Matrix m3 = m1 * 2.0;

    assert(m3[0][0] == 2.0);
    assert(m3[0][1] == 4.0);
  }

  {
    Matrix m1 = get();

    Matrix m3 = m1 / 2.0;

    assert(m3[0][0] == 0.5);
    assert(m3[0][1] == 1.0);
  }
}

auto transpose() -> void {
  Matrix m(3, 2);
  m[0][0] = 1.0;
  m[0][1] = 2.0;
  m[1][0] = 3.0;
  m[1][1] = 4.0;
  m[2][0] = 5.0;
  m[2][1] = 6.0;

  Matrix m2 = m.transpose();
  assert(m2[0][0] == 1.0);
  assert(m2[0][1] == 3.0);
  assert(m2[0][2] == 5.0);
  assert(m2[1][0] == 2.0);
  assert(m2[1][1] == 3.0);
  assert(m2[1][2] == 6.0);
}

auto serialize() -> void {
  Matrix m(4, 3);
  string serialized = m.serialize();

  assert(serialized == "0.0,0.0,0.0;0.0,0.0,0.0;0.0,0.0,0.0;0.0,0.0,0.0");
}

auto deserialize() -> void {
  auto m = Matrix::deserialize("1.2,3.2;192.293,22.11;-1.2,-3.0");

  assert(m[0][0] == 1.2);
  assert(m[0][1] == 3.2);
  assert(m[1][0] == 192.293);
  assert(m[1][1] == 22.11);
  assert(m[2][0] == -1.2);
  assert(m[2][1] == -3.0);
}

auto init() -> void {
  cout << "[matrix]" << endl;

  cout << "\t[constructor]";
  constructor();
  cout << " ✅" << endl;

  cout << "\t[randomize]";
  randomize();
  cout << " ✅" << endl;

  cout << "\t[in place operators]";
  in_place_operators();
  cout << " ✅" << endl;

  cout << "\t[global operators]";
  global_operators();
  cout << " ✅" << endl;

  cout << "\t[transposing]";
  transpose();
  cout << " ✅" << endl;

  cout << "\t[serialize]";
  serialize();
  cout << " ✅" << endl;

  cout << "\t[deserialize]";
  deserialize();
  cout << " ✅" << endl;
}

}  // namespace matrix_tests

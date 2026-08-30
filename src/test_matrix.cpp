#include "matrix.hpp"

#include <cassert>
#include <print>
#include <stdexcept>

static void
test_construction()
{
    Matrix<float> A(2, 3);
    assert(A.getRows() == 2);
    assert(A.getCols() == 3);
    assert(A(0, 0) == 0.0f);

    Matrix<float> B = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    assert(B.getRows() == 2);
    assert(B.getCols() == 3);
    assert(B(1, 2) == 6.0f);

    bool threw = false;
    try
    {
        Matrix<float> bad = {{1.0f, 2.0f}, {3.0f}};
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    assert(threw);
}

static void
test_arithmetic()
{
    Matrix<float> A = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix<float> B = {{5.0f, 6.0f}, {7.0f, 8.0f}};

    Matrix<float> sum = A + B;
    assert(sum(0, 0) == 6.0f && sum(1, 1) == 12.0f);

    Matrix<float> diff = B - A;
    assert(diff(0, 0) == 4.0f && diff(1, 1) == 4.0f);

    Matrix<float> neg = -A;
    assert(neg(0, 0) == -1.0f && neg(1, 1) == -4.0f);

    // [1 2] [5 6]   [19 22]
    // [3 4] [7 8] = [43 50]
    Matrix<float> prod = A * B;
    assert(prod(0, 0) == 19.0f);
    assert(prod(0, 1) == 22.0f);
    assert(prod(1, 0) == 43.0f);
    assert(prod(1, 1) == 50.0f);

    Matrix<float> scaled = A * 2.0f;
    assert(scaled(1, 0) == 6.0f);
    Matrix<float> scaled2 = 3.0f * A;
    assert(scaled2(1, 1) == 12.0f);

    Matrix<float> divided = A / 2.0f;
    assert(divided(0, 1) == 1.0f);
}

static void
test_compound_assign()
{
    Matrix<float> A = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix<float> B = {{1.0f, 1.0f}, {1.0f, 1.0f}};

    A += B;
    assert(A(0, 0) == 2.0f && A(1, 1) == 5.0f);

    A -= B;
    assert(A(0, 0) == 1.0f && A(1, 1) == 4.0f);

    A *= 2.0f;
    assert(A(1, 1) == 8.0f);

    A /= 2.0f;
    assert(A(1, 1) == 4.0f);
}

static void
test_equality()
{
    Matrix<float> A = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix<float> B = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix<float> C = {{1.0f, 2.0f}, {3.0f, 5.0f}};

    assert(A == B);
    assert(A != C);
}

static void
test_dimension_mismatch()
{
    Matrix<float> A(2, 3);
    Matrix<float> B(3, 2);

    bool threw = false;
    try
    {
        auto R = A + B;
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    assert(threw);

    threw = false;
    try
    {
        Matrix<float> X(2, 2);
        auto R = A * X;
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    assert(threw);
}

static void
test_division_by_zero()
{
    Matrix<float> A = {{1.0f, 2.0f}, {3.0f, 4.0f}};

    bool threw = false;
    try
    {
        auto R = A / 0.0f;
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    assert(threw);

    Matrix<float> Z = {{1.0f, 0.0f}, {2.0f, 3.0f}};
    threw = false;
    try
    {
        auto R = 1.0f / Z;
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    assert(threw);
}

static void
test_mutation()
{
    Matrix<float> A(2, 2);
    A.fill(7.0f);
    assert(A(0, 0) == 7.0f && A(1, 1) == 7.0f);

    A.resize(3, 3);
    assert(A.getRows() == 3 && A.getCols() == 3);
    assert(A(2, 2) == 0.0f);

    A.randomize(-1.0f, 1.0f);
    for (int i = 0; i < A.getRows(); ++i)
        for (int j = 0; j < A.getCols(); ++j)
            assert(A(i, j) >= -1.0f && A(i, j) <= 1.0f);

    A.clear();
    assert(A.getRows() == 0 && A.getCols() == 0);
}

static void
test_integer_randomize()
{
    Matrix<int> M(4, 4);
    M.randomize(0, 10);
    for (int i = 0; i < M.getRows(); ++i)
        for (int j = 0; j < M.getCols(); ++j)
            assert(M(i, j) >= 0 && M(i, j) <= 10);
}

static void
test_set_data()
{
    Matrix<float> A;
    A.setData({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
    assert(A.getRows() == 3 && A.getCols() == 2);
    assert(A(2, 1) == 6.0f);
}

static void
test_transpose()
{
    Matrix<float> A = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    Matrix<float> T = A.transpose();
    assert(T.getRows() == 3 && T.getCols() == 2);
    assert(T(0, 0) == 1.0f && T(0, 1) == 4.0f);
    assert(T(2, 0) == 3.0f && T(2, 1) == 6.0f);

    // (A^T)^T == A
    assert(T.transpose() == A);
}

static void
test_hadamard()
{
    Matrix<float> A = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix<float> B = {{2.0f, 0.5f}, {1.0f, 3.0f}};
    Matrix<float> H = A.hadamard(B);
    assert(H(0, 0) == 2.0f);
    assert(H(0, 1) == 1.0f);
    assert(H(1, 0) == 3.0f);
    assert(H(1, 1) == 12.0f);

    Matrix<float> C(2, 3);
    bool threw = false;
    try
    {
        auto R = A.hadamard(C);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    assert(threw);
}

static void
test_apply()
{
    Matrix<float> A = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix<float> Sq = A.apply([](float x) { return x * x; });
    assert(Sq(0, 0) == 1.0f && Sq(1, 1) == 16.0f);

    A.applyInPlace([](float x) { return x + 1.0f; });
    assert(A(0, 0) == 2.0f && A(1, 1) == 5.0f);
}

static void
test_reductions()
{
    Matrix<float> A = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    assert(A.sum() == 10.0f);
    assert(A.trace() == 5.0f);
    assert(A.isSquare());

    Matrix<float> B(2, 3);
    assert(!B.isSquare());
    bool threw = false;
    try
    {
        (void)B.trace();
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    assert(threw);
}

static void
test_row_col_access()
{
    Matrix<float> A = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    auto row0 = A.getRow(0);
    assert(row0.size() == 3 && row0[2] == 3.0f);
    auto col1 = A.getCol(1);
    assert(col1.size() == 2 && col1[0] == 2.0f && col1[1] == 5.0f);

    A.swapRows(0, 1);
    assert(A(0, 0) == 4.0f && A(1, 0) == 1.0f);
    A.swapCols(0, 2);
    assert(A(0, 0) == 6.0f && A(0, 2) == 4.0f);
}

static void
test_reshape()
{
    Matrix<float> A = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    A.reshape(3, 2);
    assert(A.getRows() == 3 && A.getCols() == 2);
    // Row-major flatten: 1,2,3,4,5,6 -> [[1,2],[3,4],[5,6]]
    assert(A(0, 0) == 1.0f && A(0, 1) == 2.0f);
    assert(A(1, 0) == 3.0f && A(1, 1) == 4.0f);
    assert(A(2, 0) == 5.0f && A(2, 1) == 6.0f);

    bool threw = false;
    try
    {
        A.reshape(4, 4);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    assert(threw);
}

static void
test_factories()
{
    auto Z = Matrix<float>::zeros(2, 3);
    assert(Z.getRows() == 2 && Z.getCols() == 3);
    assert(Z(0, 0) == 0.0f && Z(1, 2) == 0.0f);

    auto O = Matrix<float>::ones(2, 2);
    assert(O(0, 0) == 1.0f && O(1, 1) == 1.0f);

    auto I = Matrix<float>::identity(3);
    assert(I.getRows() == 3 && I.getCols() == 3);
    assert(I(0, 0) == 1.0f && I(1, 1) == 1.0f && I(2, 2) == 1.0f);
    assert(I(0, 1) == 0.0f && I(2, 0) == 0.0f);

    // Identity acts as multiplicative identity.
    Matrix<float> A = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}};
    assert(A * I == A);
}

static void
test_print()
{
    Matrix<float> A = {{1.0f, 2.0f, 3.0f}, {4.5f, 5.5f, 6.5f}};
    std::print("Default formatting:\n{}\n", A);
    std::print("Fixed width:\n{:7.2f}\n", A);

    Matrix<int> I = {{1, 2}, {3, 4}};
    std::print("Integer matrix:\n{}\n", I);
}

int
main()
{
    test_construction();
    test_arithmetic();
    test_compound_assign();
    test_equality();
    test_dimension_mismatch();
    test_division_by_zero();
    test_mutation();
    test_integer_randomize();
    test_set_data();
    test_transpose();
    test_hadamard();
    test_apply();
    test_reductions();
    test_row_col_access();
    test_reshape();
    test_factories();
    test_print();

    std::print("All tests passed.\n");
    return 0;
}

#pragma once

#include <format>
#include <initializer_list>
#include <random>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

// Row-major matrix class for storing weights and biases

template <typename T> class Matrix
{
public:
    Matrix() = default;

    Matrix(int rows, int cols)
        : rows(rows), cols(cols), data(rows, std::vector<T>(cols, T{}))
    {
    }

    Matrix(std::initializer_list<std::initializer_list<T>> initList)
        : rows(static_cast<int>(initList.size())),
          cols(initList.size() > 0
                   ? static_cast<int>(initList.begin()->size())
                   : 0)
    {
        data.reserve(rows);
        for (const auto &row : initList)
        {
            if (static_cast<int>(row.size()) != cols)
                throw std::invalid_argument(
                    "All rows in initializer list must have the same length.");
            data.emplace_back(row);
        }
    }

    friend Matrix operator+(const Matrix &a, const Matrix &b)
    {
        if (a.rows != b.rows || a.cols != b.cols)
            throw std::invalid_argument(
                "Matrix dimensions must match for addition.");

        Matrix result(a.rows, a.cols);
        for (int i = 0; i < a.rows; ++i)
            for (int j = 0; j < a.cols; ++j)
                result.data[i][j] = a.data[i][j] + b.data[i][j];
        return result;
    }

    friend Matrix operator-(const Matrix &a, const Matrix &b)
    {
        if (a.rows != b.rows || a.cols != b.cols)
            throw std::invalid_argument(
                "Matrix dimensions must match for subtraction.");

        Matrix result(a.rows, a.cols);
        for (int i = 0; i < a.rows; ++i)
            for (int j = 0; j < a.cols; ++j)
                result.data[i][j] = a.data[i][j] - b.data[i][j];
        return result;
    }

    friend Matrix operator-(const Matrix &a)
    {
        Matrix result(a.rows, a.cols);
        for (int i = 0; i < a.rows; ++i)
            for (int j = 0; j < a.cols; ++j)
                result.data[i][j] = -a.data[i][j];
        return result;
    }

    friend Matrix operator*(const Matrix &a, const Matrix &b)
    {
        if (a.cols != b.rows)
            throw std::invalid_argument(
                "Matrix dimensions must match for multiplication.");

        Matrix result(a.rows, b.cols);
        for (int i = 0; i < a.rows; ++i)
            for (int j = 0; j < b.cols; ++j)
                for (int k = 0; k < a.cols; ++k)
                    result.data[i][j] += a.data[i][k] * b.data[k][j];
        return result;
    }

    friend Matrix operator*(const Matrix &a, T scalar)
    {
        Matrix result(a.rows, a.cols);
        for (int i = 0; i < a.rows; ++i)
            for (int j = 0; j < a.cols; ++j)
                result.data[i][j] = a.data[i][j] * scalar;
        return result;
    }

    friend Matrix operator*(T scalar, const Matrix &a)
    {
        return a * scalar;
    }

    friend Matrix operator/(const Matrix &a, T scalar)
    {
        if (scalar == T{})
            throw std::invalid_argument("Division by zero.");

        Matrix result(a.rows, a.cols);
        for (int i = 0; i < a.rows; ++i)
            for (int j = 0; j < a.cols; ++j)
                result.data[i][j] = a.data[i][j] / scalar;
        return result;
    }

    friend Matrix operator/(T scalar, const Matrix &a)
    {
        Matrix result(a.rows, a.cols);
        for (int i = 0; i < a.rows; ++i)
            for (int j = 0; j < a.cols; ++j)
            {
                if (a.data[i][j] == T{})
                    throw std::invalid_argument("Division by zero.");
                result.data[i][j] = scalar / a.data[i][j];
            }
        return result;
    }

    friend bool operator==(const Matrix &a, const Matrix &b)
    {
        if (a.rows != b.rows || a.cols != b.cols)
            return false;
        for (int i = 0; i < a.rows; ++i)
            for (int j = 0; j < a.cols; ++j)
                if (a.data[i][j] != b.data[i][j])
                    return false;
        return true;
    }

    friend bool operator!=(const Matrix &a, const Matrix &b)
    {
        return !(a == b);
    }

    friend Matrix &operator+=(Matrix &a, const Matrix &b)
    {
        a = a + b;
        return a;
    }

    friend Matrix &operator-=(Matrix &a, const Matrix &b)
    {
        a = a - b;
        return a;
    }

    friend Matrix &operator*=(Matrix &a, const Matrix &b)
    {
        a = a * b;
        return a;
    }

    friend Matrix &operator*=(Matrix &a, T scalar)
    {
        a = a * scalar;
        return a;
    }

    friend Matrix &operator/=(Matrix &a, T scalar)
    {
        a = a / scalar;
        return a;
    }

    int getRows() const
    {
        return rows;
    }

    int getCols() const
    {
        return cols;
    }

    T &operator()(int row, int col)
    {
        return data[row][col];
    }

    const T &operator()(int row, int col) const
    {
        return data[row][col];
    }

    std::vector<T> &operator[](int row)
    {
        return data[row];
    }

    const std::vector<T> &operator[](int row) const
    {
        return data[row];
    }

    std::vector<std::vector<T>> &getData()
    {
        return data;
    }

    const std::vector<std::vector<T>> &getData() const
    {
        return data;
    }

    void setData(const std::vector<std::vector<T>> &newData)
    {
        data = newData;
        rows = static_cast<int>(data.size());
        cols = rows > 0 ? static_cast<int>(data[0].size()) : 0;
    }

    void setData(std::vector<std::vector<T>> &&newData)
    {
        data = std::move(newData);
        rows = static_cast<int>(data.size());
        cols = rows > 0 ? static_cast<int>(data[0].size()) : 0;
    }

    void clear()
    {
        data.clear();
        rows = 0;
        cols = 0;
    }

    void resize(int newRows, int newCols)
    {
        data.resize(newRows);
        for (auto &row : data)
            row.resize(newCols, T{});
        rows = newRows;
        cols = newCols;
    }

    void fill(T value)
    {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                data[i][j] = value;
    }

    void randomize(T minValue, T maxValue)
    {
        static thread_local std::mt19937 gen(std::random_device{}());

        using Dist = std::conditional_t<std::is_integral_v<T>,
                                        std::uniform_int_distribution<T>,
                                        std::uniform_real_distribution<T>>;
        Dist dis(minValue, maxValue);

        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                data[i][j] = dis(gen);
    }

    bool isSquare() const
    {
        return rows == cols && rows > 0;
    }

    Matrix transpose() const
    {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[j][i] = data[i][j];
        return result;
    }

    // Element-wise (Hadamard) product. Used for backprop gradient terms.
    Matrix hadamard(const Matrix &other) const
    {
        if (rows != other.rows || cols != other.cols)
            throw std::invalid_argument(
                "Matrix dimensions must match for Hadamard product.");

        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[i][j] = data[i][j] * other.data[i][j];
        return result;
    }

    // Apply a unary function to every element (useful for activations).
    template <typename Fn> Matrix apply(Fn fn) const
    {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[i][j] = fn(data[i][j]);
        return result;
    }

    template <typename Fn> void applyInPlace(Fn fn)
    {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                data[i][j] = fn(data[i][j]);
    }

    T sum() const
    {
        T s = T{};
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                s += data[i][j];
        return s;
    }

    T trace() const
    {
        if (!isSquare())
            throw std::invalid_argument(
                "Trace is only defined for square matrices.");
        T s = T{};
        for (int i = 0; i < rows; ++i)
            s += data[i][i];
        return s;
    }

    std::vector<T> getRow(int row) const
    {
        return data[row];
    }

    std::vector<T> getCol(int col) const
    {
        std::vector<T> result(rows);
        for (int i = 0; i < rows; ++i)
            result[i] = data[i][col];
        return result;
    }

    void swapRows(int a, int b)
    {
        std::swap(data[a], data[b]);
    }

    void swapCols(int a, int b)
    {
        for (int i = 0; i < rows; ++i)
            std::swap(data[i][a], data[i][b]);
    }

    // Reshape preserving total element count (row-major flatten).
    void reshape(int newRows, int newCols)
    {
        if (newRows * newCols != rows * cols)
            throw std::invalid_argument(
                "Reshape must preserve total element count.");

        std::vector<std::vector<T>> next(newRows, std::vector<T>(newCols, T{}));
        for (int k = 0; k < rows * cols; ++k)
            next[k / newCols][k % newCols] = data[k / cols][k % cols];

        data = std::move(next);
        rows = newRows;
        cols = newCols;
    }

    static Matrix zeros(int rows, int cols)
    {
        return Matrix(rows, cols);
    }

    static Matrix ones(int rows, int cols)
    {
        Matrix m(rows, cols);
        m.fill(T{1});
        return m;
    }

    static Matrix identity(int n)
    {
        Matrix m(n, n);
        for (int i = 0; i < n; ++i)
            m.data[i][i] = T{1};
        return m;
    }

private:
    int rows = 0;
    int cols = 0;
    std::vector<std::vector<T>> data;
};

// std::formatter specialization so Matrix<T> works with std::format /
// std::print (C++23). Forwards an optional element format-spec to each cell:
//   std::print("{}", m);        // default formatting
//   std::print("{:.3f}", m);    // apply ".3f" to each element
template <typename T, typename CharT>
struct std::formatter<Matrix<T>, CharT>
{
    std::basic_string<CharT> elemSpec;

    constexpr auto parse(std::basic_format_parse_context<CharT> &ctx)
    {
        auto it  = ctx.begin();
        auto end = ctx.end();
        elemSpec.clear();
        while (it != end && *it != CharT('}'))
            elemSpec.push_back(*it++);
        return it;
    }

    template <typename FormatContext>
    auto format(const Matrix<T> &m, FormatContext &ctx) const
    {
        auto out = ctx.out();
        auto fmtStr =
            std::basic_string<CharT>{CharT('{'), CharT(':')} + elemSpec +
            std::basic_string<CharT>{CharT('}')};
        std::basic_string_view<CharT> fmtView{fmtStr};

        out = std::format_to(out, "[");
        for (int i = 0; i < m.getRows(); ++i)
        {
            out = std::format_to(out, "[");
            for (int j = 0; j < m.getCols(); ++j)
            {
                out = std::vformat_to(
                    out, fmtView,
                    std::make_format_args<std::basic_format_context<
                        decltype(out), CharT>>(m(i, j)));
                if (j + 1 < m.getCols())
                    out = std::format_to(out, ", ");
            }
            out = std::format_to(out, "]");
            if (i + 1 < m.getRows())
                out = std::format_to(out, "\n ");
        }
        return std::format_to(out, "]");
    }
};

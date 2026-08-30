#pragma once

#include "matrix.hpp"

#include <memory>

enum class OperationType
{
    None,
    Add,
    Subtract,
    Multiply,
    Divide,
    MatMul,
    Transpose,
    Reshape,
    Sum,
    Mean,
    Max,
    Min
};

template <typename T>
class Tensor : public std::enable_shared_from_this<Tensor<T>>
{
    using Parents = std::vector<std::shared_ptr<Tensor<T>>>;

public:
    Tensor(const Matrix<T> &data, bool requires_grad = false,
           OperationType op_type = OperationType::None, Parents parents = {});

    void backward(const Matrix<T> &grad = Matrix<T>());

    inline const Matrix<T> &get_data() const
    {
        return data;
    }

    inline const Matrix<T> &get_grad() const
    {
        return grad_buffer;
    }

    inline bool is_requires_grad() const
    {
        return requires_grad;
    }

private:
    Matrix<T> data;
    bool requires_grad;
    Matrix<T> grad_buffer;
    OperationType op_type;
    Parents parents;
};

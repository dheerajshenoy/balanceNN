#include "tensor.hpp"

template <typename T>
Tensor<T>::Tensor(const Matrix<T> &data, bool requires_grad,
                  OperationType op_type, Parents parents)
{
}

template <typename T>
void
Tensor<T>::backward(const Matrix<T> &grad)
{
}

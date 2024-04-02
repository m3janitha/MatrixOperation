#pragma once

#include <matrix_operations/matrix_impl.h>

namespace matrix
{
    template <std::size_t Rows, std::size_t Columns>
    using Matrix = MatrixImpl<double, Rows, Columns>;
}

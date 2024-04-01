#pragma once

#include "matrix_impl_copy_2.h"
// #include "matrix_multi_naive_copy.h"
// #include "matrix_multi_optimized _t_1_copy.h"
// #include "matrix_multi_optimized _t_n_copy.h"

namespace matrix
{ 
    template <std::size_t Rows, std::size_t Columns>
    using Matrix = MatrixImpl<double, Rows, Columns>;
}

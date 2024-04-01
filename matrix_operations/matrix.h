#pragma once

#include "matrix_multi_naive.h"
#include "matrix_multi_optimized _t_1.h"
#include "matrix_multi_optimized _t_n.h"

namespace matrix
{
    template <std::size_t Rows, std::size_t Columns>
    using Matrix = MatrixMultNaive<double, Rows, Columns>;

    template <std::size_t Rows, std::size_t Columns>
    using MatrixT1 = MatrixOptimizedT1<double, Rows, Columns>;

    template <std::size_t Rows, std::size_t Columns>
    using MatrixTN = MatrixOptimizedTN<double, Rows, Columns>;

    template <std::size_t R, std::size_t C>
    MatrixT1<R, C> ab_c(MatrixT1<R, C> &a, MatrixT1<R, C> &b, MatrixT1<R, C> &c) noexcept
    {
        MatrixT1<R, C> result{};
        /* For each row in A */
        for (std::size_t i{0}; i < a.rows(); i++)
        {
            /* For each column in A (row in B) */
            for (std::size_t k{0}; k < a.columns(); k++)
            {
                auto data_ik = a.data_[i][k];
                /* For each column in B (column in R) */
                for (std::size_t j{0}; j < b.columns(); j++)
                {
                    result.data_[i][j] += data_ik * b.data_[k][j];
                }
            }
            for (std::size_t k{0}; k < a.columns(); k++)
            {
                result.data_[i][k] = result.data_[i][k] + c.data_[i][k];
            }
        }
        return result;
    }
}

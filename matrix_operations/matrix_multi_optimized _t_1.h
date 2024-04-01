#pragma once

#include "matrix_impl.h"
#include <array>
#include <cstddef>
#include <iostream>
namespace matrix
{
    /* Cache optimised single threaded (t1) implementation */
    template <typename T, std::size_t Rows, std::size_t Columns>
    class MatrixOptimizedT1 : public MatrixImpl<MatrixOptimizedT1<T, Rows, Columns>, T, Rows, Columns>
    {
    public:
        constexpr MatrixOptimizedT1() = default;
        constexpr explicit MatrixOptimizedT1(const std::array<std::array<T, Columns>, Rows> &data) : MatrixImpl<MatrixOptimizedT1<T, Rows, Columns>, T, Rows, Columns>(data){};

        /* import all multiplication operators */
        using MatrixImpl<MatrixOptimizedT1<T, Rows, Columns>, T, Rows, Columns>::operator*;

        template <std::size_t OtherColumns>
        constexpr MatrixOptimizedT1<T, Rows, OtherColumns> operator*(const MatrixOptimizedT1<T, Columns, OtherColumns> &other) const noexcept;

        template <std::size_t OtherColumns>
        constexpr MatrixOptimizedT1<T, Rows, OtherColumns> multiplication(const MatrixOptimizedT1<T, Rows, Columns>& lhs, const MatrixOptimizedT1<T, Columns, OtherColumns> &other) const noexcept;
    };

    /* A . B = R */
    /* Read single value from Matrix A at once and cache it (in register) */
    /* For rows in A, For Columns in A(row in B), for Columns in B */
    /* B and R are traversed by rows to improve cache coherence */

    template <typename T, std::size_t Rows, std::size_t Columns>
    template <std::size_t OtherColumns>
    constexpr MatrixOptimizedT1<T, Rows, OtherColumns> MatrixOptimizedT1<T, Rows, Columns>::operator*(const MatrixOptimizedT1<T, Columns, OtherColumns> &other) const noexcept
    {
        MatrixOptimizedT1<T, Rows, OtherColumns> result{};
        /* For each row in A */
        for (std::size_t i{0}; i < this->rows(); i++)
        {
            /* For each column in A (row in B) */
            for (std::size_t k{0}; k < this->columns(); k++)
            {
                auto data_ik = this->data_[i][k];
                /* For each column in B (column in R) */
                for (std::size_t j{0}; j < other.columns(); j++)
                {
                    result.data_[i][j] += data_ik * other.data_[k][j];
                }
            }
        }
        return result;
    }

    template <typename T, std::size_t Rows, std::size_t Columns>
    template <std::size_t OtherColumns>
    constexpr MatrixOptimizedT1<T, Rows, OtherColumns> MatrixOptimizedT1<T, Rows, Columns>::multiplication(const MatrixOptimizedT1<T, Rows, Columns>& lhs, const MatrixOptimizedT1<T, Columns, OtherColumns> &rhs) const noexcept
    {
        MatrixOptimizedT1<T, Rows, OtherColumns> result{};
        /* For each row in A */
        for (std::size_t i{0}; i < lhs.rows(); i++)
        {
            /* For each column in A (row in B) */
            for (std::size_t k{0}; k < lhs.columns(); k++)
            {
                auto data_ik = lhs.data_[i][k];
                /* For each column in B (column in R) */
                for (std::size_t j{0}; j < rhs.columns(); j++)
                {
                    result.data_[i][j] += data_ik * rhs.data_[k][j];
                }
            }
        }
        return result;
    }
}

#pragma once

#include <array>
#include <cstddef>
#include <iostream>
namespace matrix
{
    /* The naive approach */
    template <typename T, std::size_t Rows, std::size_t Columns>
    class MatrixMultNaive
    {
    public:
        template <std::size_t OtherColumns>
        constexpr MatrixMultNaive<T, Rows, OtherColumns> multiplication(const MatrixMultNaive<T, Rows, Columns>& lhs, const MatrixMultNaive<T, Columns, OtherColumns> &rhs) const noexcept;
    };

    /* Not cache friendly */
    template <typename T, std::size_t Rows, std::size_t Columns>
    template <std::size_t OtherColumns>
    constexpr MatrixMultNaive<T, Rows, OtherColumns> MatrixMultNaive<T, Rows, Columns>::multiplication(const MatrixMultNaive<T, Rows, Columns>& lhs, const MatrixMultNaive<T, Columns, OtherColumns> &rhs) const noexcept
    {
        MatrixMultNaive<T, Rows, OtherColumns> result{};
        for (std::size_t row{0}; row < lhs.rows(); row++)
        {
            for (std::size_t other_col{0}; other_col < rhs.columns(); other_col++)
            {
                T sum{0};
                for (std::size_t column{0}; column < lhs.columns(); column++)
                {
                    sum += lhs.data_[row][column] * rhs.data_[column][other_col];
                }
                result.data_[row][other_col] = sum;
            }
        }

        return result;
    }
}

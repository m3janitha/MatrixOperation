#pragma once

#include "matrix.h"

namespace matrix
{
    template <typename T, std::size_t Rows, std::size_t Columns, std::size_t OtherColumns>
    inline /*constexpr*/ MatrixImpl<T, Rows, OtherColumns> ab_c_generic(MatrixImpl<T, Rows, Columns> &a, MatrixImpl<T, Rows, OtherColumns> &b, MatrixImpl<T, Rows, OtherColumns> &c)
    {
        return a * b + c;
    }

    template <typename T, std::size_t Rows, std::size_t Columns, std::size_t OtherColumns>
    constexpr void ab_c_optimised_aux(MatrixImpl<T, Rows, OtherColumns> &result, MatrixImpl<T, Rows, Columns> &a, MatrixImpl<T, Rows, OtherColumns> &b, MatrixImpl<T, Rows, OtherColumns> &c, std::size_t start, std::size_t end)
    {
        for (std::size_t i{start}; i < end; i++)
        {
            /* For each column in A (row in B) */
            for (std::size_t k{0}; k < a.columns(); k++)
            {
                auto data_ik = a.data()[i][k];
                /* For each column in B (column in R) */
                for (std::size_t j{0}; j < b.columns(); j++)
                {
                    result.data()[i][j] += data_ik * b.data()[k][j];
                }
            }
            /* AB + C for i th row.*/
            for (std::size_t k{0}; k < a.columns(); k++)
            {
                result.data()[i][k] = result.data()[i][k] + c.data()[i][k];
            }
        }
    }

    template <typename T, std::size_t Rows, std::size_t Columns, std::size_t OtherColumns>
    inline constexpr MatrixImpl<T, Rows, OtherColumns> ab_c_optimised(MatrixImpl<T, Rows, Columns> &a, MatrixImpl<T, Rows, OtherColumns> &b, MatrixImpl<T, Rows, OtherColumns> &c)
    {
        MatrixImpl<T, Rows, OtherColumns> result{};
        ab_c_optimised_aux(result, a, b, c, 0, a.rows());
        return result;
    }

    template <typename T, std::size_t Rows, std::size_t Columns, std::size_t OtherColumns>
    inline /*constexpr*/ MatrixImpl<T, Rows, OtherColumns> ab_c_optimised_tn(MatrixImpl<T, Rows, Columns> &a, MatrixImpl<T, Rows, OtherColumns> &b, MatrixImpl<T, Rows, OtherColumns> &c)
    {
        MatrixImpl<T, Rows, OtherColumns> result{};
        std::vector<std::thread> threads{};
        for (auto &chunk : MatrixImpl<T, Rows, Columns>::chunks_)
        {
            /* thread creation is costly, use thread pools for better performance */
            threads.emplace_back([&result, &a, &b, &c, &chunk]()
                                 { ab_c_optimised_aux(result, a, b, c, chunk.first, chunk.second); });
        }

        for (auto &thread : threads)
        {
            thread.join();
        }

        return result;
    }
}

#pragma once

#include "matrix.h"

namespace matrix
{
    /* Reusing Matrix class operators */
    template <typename T, std::size_t Rows, std::size_t Columns, std::size_t OtherColumns>
    inline /*constexpr*/ MatrixImpl<T, Rows, OtherColumns> ab_c_generic(MatrixImpl<T, Rows, Columns> &a, MatrixImpl<T, Rows, OtherColumns> &b, MatrixImpl<T, Rows, OtherColumns> &c)
    {
        return a * b + c;
    }

    /* A . B  + C = R */
    /* Read single value from Matrix A at once and cache it (in register) */
    /* For rows in A, For Columns in A, for Columns in B */
    /* B and R are traversed by rows to improve cache coherence */
    /* Compute A . B and add C after each row is completed */
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

    /* single threaded (t1) implementation */
    template <typename T, std::size_t Rows, std::size_t Columns, std::size_t OtherColumns>
    inline constexpr MatrixImpl<T, Rows, OtherColumns> ab_c_optimised(MatrixImpl<T, Rows, Columns> &a, MatrixImpl<T, Rows, OtherColumns> &b, MatrixImpl<T, Rows, OtherColumns> &c)
    {
        MatrixImpl<T, Rows, OtherColumns> result{};
        ab_c_optimised_aux(result, a, b, c, 0, a.rows());
        return result;
    }

    /* multi threaded (tn) implementation */
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

#pragma once

#include <matrix_operations/matrix.h>

namespace matrix
{
    /* Reusing Matrix class operators */
    template <typename T, std::size_t Rows, std::size_t Columns, std::size_t OtherColumns>
    inline /*constexpr*/ MatrixImpl<T, Rows, OtherColumns> ab_c_generic(MatrixImpl<T, Rows, Columns> &a, MatrixImpl<T, Columns, OtherColumns> &b, MatrixImpl<T, Rows, OtherColumns> &c)
    {
        return (a * b) + c;
    }

    /* A . B  + C = R */
    /* Read single value from Matrix A at once and cache it (in register) */
    /* For rows in A, For Columns in A, for Columns in B */
    /* B and R are traversed by rows to improve cache coherence */
    /* Compute A . B and add C after each row is completed */
    template <typename T, std::size_t Rows, std::size_t Columns, std::size_t OtherColumns>
    constexpr void ab_c_optimised_aux(MatrixImpl<T, Rows, OtherColumns> &result, MatrixImpl<T, Rows, Columns> &a, MatrixImpl<T, Columns, OtherColumns> &b, MatrixImpl<T, Rows, OtherColumns> &c, std::size_t start, std::size_t end)
    {
        /* For each row in A from start to end */
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
            for (std::size_t j{0}; j < b.columns(); j++)
            {
                result.data()[i][j] = result.data()[i][j] + c.data()[i][j];
            }
        }
    }

    /* single threaded (t1) implementation */
    template <typename T, std::size_t Rows, std::size_t Columns, std::size_t OtherColumns>
    inline constexpr MatrixImpl<T, Rows, OtherColumns> ab_c_optimised(MatrixImpl<T, Rows, Columns> &a, MatrixImpl<T, Columns, OtherColumns> &b, MatrixImpl<T, Rows, OtherColumns> &c)
    {
        MatrixImpl<T, Rows, OtherColumns> result{};
        ab_c_optimised_aux(result, a, b, c, 0, a.rows());
        return result;
    }

    /* multi threaded (tn) implementation */
    template <typename T, std::size_t Rows, std::size_t Columns, std::size_t OtherColumns>
    inline /*constexpr*/ MatrixImpl<T, Rows, OtherColumns> ab_c_optimised_tn(MatrixImpl<T, Rows, Columns> &a, MatrixImpl<T, Columns, OtherColumns> &b, MatrixImpl<T, Rows, OtherColumns> &c)
    {
        MatrixImpl<T, Rows, OtherColumns> result{};
        std::vector<std::thread> threads{};
        for (auto &chunk : MatrixImpl<T, Rows, Columns>::get_chunks())
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

    template <typename T, std::size_t Rows, std::size_t Columns, std::size_t OtherColumns>
    inline /*constexpr*/ MatrixImpl<T, Rows, OtherColumns> ab_c_omp(MatrixImpl<T, Rows, Columns> &a, MatrixImpl<T, Columns, OtherColumns> &b, MatrixImpl<T, Rows, OtherColumns> &c)
    {
        MatrixImpl<T, Rows, OtherColumns> result{};
        std::size_t i{0};
        std::size_t j{0};
        std::size_t k{0};
        omp_set_num_threads(MatrixImpl<T, Rows, Columns>::number_of_worker_threads());
#pragma omp parallel for private(i, j, k)
        for (i = 0; i < a.rows(); i++)
        {
            /* For each column in A (row in B) */
            for (k = 0; k < a.columns(); k++)
            {
                auto data_ik = a.data()[i][k];
                /* For each column in B (column in R) */
                for (j = 0; j < b.columns(); j++)
                {
                    result.data()[i][j] += data_ik * b.data()[k][j];
                }
            }
            /* AB + C for i th row.*/
            for (j = 0; j < b.columns(); j++)
            {
                result.data()[i][j] = result.data()[i][j] + c.data()[i][j];
            }
        }

        return result;
    }

    /* Final */
    template <typename T, std::size_t Rows, std::size_t Columns, std::size_t OtherColumns>
    constexpr MatrixImpl<T, Rows, OtherColumns> ab_c(MatrixImpl<T, Rows, Columns> &a, MatrixImpl<T, Columns, OtherColumns> &b, MatrixImpl<T, Rows, OtherColumns> &c)
    {
        /* points to the best solution */
        if constexpr (Rows * Columns * OtherColumns < 8 * 8 * 8)
            return ab_c_optimised(a, b, c);
        else if constexpr (Rows * Columns * OtherColumns < 128 * 128 * 128)
            return ab_c_omp(a, b, c);
        else
            return ab_c_optimised_tn(a, b, c);
    }
}

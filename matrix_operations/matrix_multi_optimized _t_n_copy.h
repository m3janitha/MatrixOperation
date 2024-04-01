#pragma once

#include "matrix_impl.h"
#include <array>
#include <thread>
#include <cstddef>

namespace matrix
{
    /* Cache optimised multi threaded (tn) implementation */
    template <typename T, std::size_t Rows, std::size_t Columns>
    class MatrixOptimizedTN
    {
    public:
        template <std::size_t OtherColumns>
        constexpr MatrixOptimizedTN<T, Rows, OtherColumns> multiplication(const MatrixOptimizedTN<T, Rows, Columns>& lhs, const MatrixOptimizedTN<T, Columns, OtherColumns> &rhs) const noexcept;


    private:
        template <std::size_t OtherColumns>
        constexpr void multiplication_parallel_aux(MatrixOptimizedTN<T, Rows, OtherColumns> &result, const MatrixOptimizedTN<T, Rows, Columns>& lhs, const MatrixOptimizedTN<T, Columns, OtherColumns> &rhs, std::size_t start, std::size_t end) const noexcept;

        using Chunks = std::vector<std::pair<std::size_t, std::size_t>>;
        static Chunks compute_parallel_chunks(const std::size_t array_length, const std::size_t number_of_threads);

        /* This should ideally set from config based on the host arch */
        static constexpr std::size_t number_of_worker_threads{8};
        
        /* Chunks are calcluated once for this class */
        inline static const Chunks chunks_{compute_parallel_chunks(MatrixImpl<MatrixOptimizedTN<T, Rows, Columns>, T, Rows, Columns>::rows(), number_of_worker_threads)};
    };

    /* Compute chunks for each worker thread. This is done only once per this class */
    template <typename T, std::size_t Rows, std::size_t Columns>
    MatrixOptimizedTN<T, Rows, Columns>::Chunks MatrixOptimizedTN<T, Rows, Columns>::compute_parallel_chunks(const std::size_t array_length, const std::size_t number_of_threads)
    {
        std::size_t number_of_chunks = number_of_threads;
        std::size_t chunk_size = array_length / number_of_chunks;
        std::size_t remainder = array_length % number_of_threads;

        Chunks chunks{};
        std::size_t start_index{0};
        for (std::size_t i{0}; i < number_of_chunks; i++)
        {
            auto chunk_length = chunk_size + (i < remainder ? 1 : 0);
            chunks.emplace_back(start_index, start_index + chunk_length);
            start_index += chunk_length;
        }
        return chunks;
    }

    /* A . B = R */
    /* Read single value from Matrix A at once and cache it (in register) */
    /* For rows in A, For Columns in A(row in B), for Columns in B */
    /* B and R are traversed by rows to improve cache coherence */

    template <typename T, std::size_t Rows, std::size_t Columns>
    template <std::size_t OtherColumns>
    constexpr void MatrixOptimizedTN<T, Rows, Columns>::multiplication_parallel_aux(MatrixOptimizedTN<T, Rows, OtherColumns> &result, const MatrixOptimizedTN<T, Rows, Columns>& lhs, const MatrixOptimizedTN<T, Columns, OtherColumns> &other, std::size_t start, std::size_t end) const noexcept
    {
        /* For each row in A from Start to End of this chunk */
        for (std::size_t i{start}; i < end; i++)
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
    }

    /* Thread creation on demand is costly. Use the thread pool */
    /* Run multiplication_parallel_aux for every chunk in a different thread */
    /* Run threads on isolated CPUs for better performance */
    /* Use CPUs on a single NUMA node. Cross NUMA memory access is expensive */
    template <typename T, std::size_t Rows, std::size_t Columns>
    template <std::size_t OtherColumns>
    constexpr MatrixOptimizedTN<T, Rows, OtherColumns> MatrixOptimizedTN<T, Rows, Columns>::multiplication(const MatrixOptimizedTN<T, Rows, Columns>& lhs, const MatrixOptimizedTN<T, Columns, OtherColumns> &rhs) const noexcept
    {
        MatrixOptimizedTN<T, Rows, OtherColumns> result{};
        std::vector<std::thread> threads{};
        for (auto &chunk : chunks_)
        {
            /* thread creation is costly, use thread pools for better performance */
            threads.emplace_back([&result, &lhs, &rhs, &chunk]()
                                 { multiplication_parallel_aux(result, lhs, rhs, chunk.first, chunk.second); });
        }

        for (auto &thread : threads)
        {
            thread.join();
        }

        return result;
    }
}

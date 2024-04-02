#pragma once

#include <array>
#include <vector>
#include <cstddef>
#include <iostream>
#include <thread>
#include <omp.h>

namespace matrix
{
    template <typename T, std::size_t Rows, std::size_t Columns>
    class MatrixImpl
    {
    public:
        using Data = std::array<std::array<T, Columns>, Rows>;

        constexpr MatrixImpl() = default;
        constexpr explicit MatrixImpl(const Data &data) : data_(data){};

        /* Public getters */
        static constexpr std::size_t rows() noexcept { return Rows; }
        static constexpr std::size_t columns() noexcept { return Columns; }
        using Chunks = std::vector<std::pair<std::size_t, std::size_t>>;
        static const Chunks &get_chunks() { return chunks_; }
        static std::size_t number_of_worker_threads() { return number_of_worker_threads_; }

        /* Getter for data */
        constexpr Data &data() { return data_; }
        constexpr const Data &data() const { return data_; }

        /* Scalar multiplication (matrix * scalar) */
        constexpr MatrixImpl operator*(T scalar) const & noexcept;
        /* Scalar multiplication (matrix * scalar) (reuse the rvalue instead of allocatiing new) */
        constexpr MatrixImpl operator*(T scalar) && noexcept;

        /* Scalar multiplication (scalar * matrix) */
        friend constexpr MatrixImpl operator*(T scalar, const MatrixImpl &mat) noexcept { return mat * scalar; }
        /* Scalar multiplication (scalar * matrix) (reuse the rvalue instead of allocatiing new) */
        friend constexpr MatrixImpl operator*(T scalar, MatrixImpl &&mat) noexcept { return std::move(mat) * scalar; }

        template <std::size_t OtherColumns>
        constexpr MatrixImpl<T, Rows, OtherColumns> operator*(const MatrixImpl<T, Columns, OtherColumns> &other) const noexcept;

        /* Addition for lvlues */
        constexpr MatrixImpl operator+(const MatrixImpl &other) const & noexcept;
        /* Addition for rvalues (reuse the rvalue instead of allocatiing new) */
        constexpr MatrixImpl operator+(const MatrixImpl &other) && noexcept;

        /* Subtraction for lvlues */
        constexpr MatrixImpl operator-(const MatrixImpl &other) const & noexcept;
        /* Subtraction for rvalues (reuse the rvalue instead of allocatiing new) */
        constexpr MatrixImpl operator-(const MatrixImpl &other) && noexcept;

        /* Different multiplication implementations (public for user convenience) */
        template <std::size_t OtherColumns>
        constexpr MatrixImpl<T, Rows, OtherColumns> multiplication_naive(const MatrixImpl<T, Columns, OtherColumns> &other) const noexcept;

        /* Cache optimised single threaded (t1) implementation */
        template <std::size_t OtherColumns>
        constexpr MatrixImpl<T, Rows, OtherColumns> multiplication_t1(const MatrixImpl<T, Columns, OtherColumns> &other) const noexcept;

        /* Cache optimised multi threaded (tn) implementation */
        template <std::size_t OtherColumns>
        MatrixImpl<T, Rows, OtherColumns> multiplication_tn(const MatrixImpl<T, Columns, OtherColumns> &other) const noexcept;

        template <std::size_t OtherColumns>
        constexpr void multiplication_t_aux(MatrixImpl<T, Rows, OtherColumns> &result, const MatrixImpl<T, Columns, OtherColumns> &other, std::size_t start, std::size_t end) const noexcept;

        /* Cache optimised blocked (t1) implementation */
        template <std::size_t OtherColumns>
        constexpr MatrixImpl<T, Rows, OtherColumns> multiplication_blocked(const MatrixImpl<T, Columns, OtherColumns> &other) const noexcept;

        /* Cache optimised blocked (t1) implementation */
        template <std::size_t OtherColumns>
        constexpr MatrixImpl<T, Rows, OtherColumns> multiplication_omp(const MatrixImpl<T, Columns, OtherColumns> &other) const noexcept;

        /* single threaded (t1) implementation */
        constexpr MatrixImpl addition(const MatrixImpl &other) const & noexcept;
        /* single threaded (t1) implementation for rvalues */
        constexpr MatrixImpl addition(const MatrixImpl &other) && noexcept;

        /* multi threaded (tn) implementation */
        MatrixImpl addition_tn(const MatrixImpl &other) const & noexcept;
        /* multi threaded (tn) implementation for rvalues */
        MatrixImpl addition_tn(const MatrixImpl &other) && noexcept;

        auto operator<=>(const MatrixImpl &) const = default;

    private:
        constexpr void addition_tn_aux(MatrixImpl &result, const MatrixImpl &other, std::size_t start, std::size_t end) const noexcept;
        constexpr void addition_tn_aux(const MatrixImpl &other, std::size_t start, std::size_t end) noexcept;
        static Chunks compute_parallel_chunks(const std::size_t array_length, const std::size_t number_of_threads);

        /* This should ideally set from config based on the host arch */
        static constexpr std::size_t number_of_worker_threads_{8};
        /* Chunks are calcluated once for this class */
        inline static const Chunks chunks_{compute_parallel_chunks(MatrixImpl::rows(), number_of_worker_threads_)};
        /* L1 chache line size */
        static constexpr std::size_t hardware_constructive_interference_size{32};
        static constexpr std::size_t block_size_{hardware_constructive_interference_size / sizeof(T)};
        /* Data */
        std::array<std::array<T, Columns>, Rows> data_{};
    };

    template <typename T, std::size_t Rows, std::size_t Columns>
    inline std::ostream &operator<<(std::ostream &os, const MatrixImpl<T, Rows, Columns> &matrix)
    {
        for (std::size_t row{0}; row < matrix.rows(); row++)
        {
            os << "[ ";
            for (std::size_t column{0}; column < matrix.columns(); column++)
            {
                os << matrix.data()[row][column] << " ";
            }
            os << "]" << std::endl;
        }
        return os;
    }

    template <typename T, std::size_t Rows, std::size_t Columns>
    constexpr MatrixImpl<T, Rows, Columns> MatrixImpl<T, Rows, Columns>::operator*(T scalar) const & noexcept
    {
        MatrixImpl result{};
        for (std::size_t row{0}; row < rows(); row++)
        {
            for (std::size_t column{0}; column < columns(); column++)
            {
                result.data_[row][column] = data_[row][column] * scalar;
            }
        }
        return result;
    }

    template <typename T, std::size_t Rows, std::size_t Columns>
    constexpr MatrixImpl<T, Rows, Columns> MatrixImpl<T, Rows, Columns>::operator*(T scalar) && noexcept
    {
        for (std::size_t row{0}; row < rows(); row++)
        {
            for (std::size_t column{0}; column < columns(); column++)
            {
                data_[row][column] *= scalar;
            }
        }
        return *this;
    }

    template <typename T, std::size_t Rows, std::size_t Columns>
    template <std::size_t OtherColumns>
    constexpr MatrixImpl<T, Rows, OtherColumns> MatrixImpl<T, Rows, Columns>::operator*(const MatrixImpl<T, Columns, OtherColumns> &other) const noexcept
    {
        if constexpr (Rows * Columns * OtherColumns < 8 * 8 * 8)
            return multiplication_naive(other);
        else if constexpr (Rows * Columns * OtherColumns < 128 * 128 * 128)
            return multiplication_omp(other);
        else
            return multiplication_tn(other);
    }

    /* Not cache friendly */
    template <typename T, std::size_t Rows, std::size_t Columns>
    template <std::size_t OtherColumns>
    constexpr MatrixImpl<T, Rows, OtherColumns> MatrixImpl<T, Rows, Columns>::multiplication_naive(const MatrixImpl<T, Columns, OtherColumns> &other) const noexcept
    {
        MatrixImpl<T, Rows, OtherColumns> result{};
        for (std::size_t row{0}; row < rows(); row++)
        {
            for (std::size_t other_col{0}; other_col < other.columns(); other_col++)
            {
                T sum{0};
                for (std::size_t column{0}; column < columns(); column++)
                {
                    sum += data_[row][column] * other.data()[column][other_col];
                }
                result.data()[row][other_col] = sum;
            }
        }

        return result;
    }

    /* A . B = R */
    /* Read single value from Matrix A at once and cache it (in register) */
    /* For rows in A, For Columns in A(row in B), for Columns in B */
    /* B and R are traversed by rows to improve cache coherence */
    template <typename T, std::size_t Rows, std::size_t Columns>
    template <std::size_t OtherColumns>
    constexpr void MatrixImpl<T, Rows, Columns>::multiplication_t_aux(MatrixImpl<T, Rows, OtherColumns> &result, const MatrixImpl<T, Columns, OtherColumns> &other, std::size_t start, std::size_t end) const noexcept
    {
        /* For each row in A from Start to End of this chunk */
        for (std::size_t i{start}; i < end; i++)
        {
            /* For each column in A (row in B) */
            for (std::size_t k{0}; k < columns(); k++)
            {
                auto data_ik = data_[i][k];
                /* For each column in B (column in R) */
                for (std::size_t j{0}; j < other.columns(); j++)
                {
                    result.data()[i][j] += data_ik * other.data()[k][j];
                }
            }
        }
    }

    template <typename T, std::size_t Rows, std::size_t Columns>
    template <std::size_t OtherColumns>
    constexpr MatrixImpl<T, Rows, OtherColumns> MatrixImpl<T, Rows, Columns>::multiplication_t1(const MatrixImpl<T, Columns, OtherColumns> &other) const noexcept
    {
        MatrixImpl<T, Rows, OtherColumns> result{};
        multiplication_t_aux(result, other, 0, rows());
        return result;
    }

    /* Compute chunks for each worker thread. This is done only once per this type */
    template <typename T, std::size_t Rows, std::size_t Columns>
    typename MatrixImpl<T, Rows, Columns>::Chunks MatrixImpl<T, Rows, Columns>::compute_parallel_chunks(const std::size_t array_length, const std::size_t number_of_threads)
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

    /* Thread creation on demand is costly. Consider using a thread pool instead. */
    /* Execute multiplication_t_aux for each chunk in a separate thread. */
    /* Run threads on isolated CPUs for better performance. */
    /* Utilize CPUs within a single NUMA node. Cross-NUMA memory access is expensive. */
    template <typename T, std::size_t Rows, std::size_t Columns>
    template <std::size_t OtherColumns>
    MatrixImpl<T, Rows, OtherColumns> MatrixImpl<T, Rows, Columns>::multiplication_tn(const MatrixImpl<T, Columns, OtherColumns> &other) const noexcept
    {
        MatrixImpl<T, Rows, OtherColumns> result{};
        /* jthread is not supported in Clang 14 yet, so std::thread is being used instead. */
        std::vector<std::thread> threads{};
        for (auto &chunk : chunks_)
        {
            /* thread creation is costly, use thread pools for better performance */
            threads.emplace_back([this, &result, &other, &chunk]()
                                 { multiplication_t_aux(result, other, chunk.first, chunk.second); });
        }

        for (auto &thread : threads)
        {
            thread.join();
        }

        return result;
    }

    /* This implimentation seems to have a bug */
    template <typename T, std::size_t Rows, std::size_t Columns>
    template <std::size_t OtherColumns>
    constexpr MatrixImpl<T, Rows, OtherColumns> MatrixImpl<T, Rows, Columns>::multiplication_blocked(const MatrixImpl<T, Columns, OtherColumns> &other) const noexcept
    {
        static_assert(Rows == Columns);
        static_assert(Rows == OtherColumns);
        MatrixImpl<T, Rows, OtherColumns> result{};
        /* For each row in A */
        for (std::size_t i = 0; i < rows(); i++)
        {
            /* For each column in A (row in B), advance by Block Size */
            for (std::size_t i_block = 0; i_block < columns(); i_block += block_size_)
            {
                // For each chunk of A/B for this block
                for (std::size_t k = 0; k < other.columns(); k += block_size_)
                {
                    // For each row in the chunk
                    for (std::size_t k_block = 0; k_block < block_size_; k_block++)
                    {
                        // Go through all the elements in the sub chunk
                        for (std::size_t idx = 0; idx < block_size_; idx++)
                        {
                            result.data()[i][i_block + idx] += data()[i][k + k_block] * other.data()[k + k_block][i_block + idx];
                        }
                    }
                }
            }
        }

        return result;
    }

    template <typename T, std::size_t Rows, std::size_t Columns>
    template <std::size_t OtherColumns>
    constexpr MatrixImpl<T, Rows, OtherColumns> MatrixImpl<T, Rows, Columns>::multiplication_omp(const MatrixImpl<T, Columns, OtherColumns> &other) const noexcept
    {
        MatrixImpl<T, Rows, OtherColumns> result{};
        std::size_t i{0};
        std::size_t j{0};
        std::size_t k{0};
        omp_set_num_threads(number_of_worker_threads_);
        #pragma omp parallel for private(i, j, k)
        for (i = 0; i < rows(); i++)
        {
            /* For each column in A (row in B) */
            // #pragma omp parallel for
            for (k = 0; k < columns(); k++)
            {
                /* For each column in B (column in R) */
                // #pragma omp parallel for
                for (j = 0; j < other.columns(); j++)
                {
                    result.data()[i][j] += data_[i][k] * other.data()[k][j];
                }
            }
        }
        return result;
    }

    template <typename T, std::size_t Rows, std::size_t Columns>
    constexpr MatrixImpl<T, Rows, Columns> MatrixImpl<T, Rows, Columns>::operator+(const MatrixImpl &other) const & noexcept
    {
        return addition(other);
    }

    template <typename T, std::size_t Rows, std::size_t Columns>
    constexpr MatrixImpl<T, Rows, Columns> MatrixImpl<T, Rows, Columns>::operator+(const MatrixImpl &other) && noexcept
    {
        return addition(other);
    }

    template <typename T, std::size_t Rows, std::size_t Columns>
    constexpr void MatrixImpl<T, Rows, Columns>::addition_tn_aux(MatrixImpl &result, const MatrixImpl &other, std::size_t start, std::size_t end) const noexcept
    {
        /* For each row in A from Start to End of this chunk */
        for (std::size_t row{start}; row < end; row++)
        {
            for (std::size_t column{0}; column < columns(); column++)
            {
                result.data_[row][column] = data_[row][column] + other.data_[row][column];
            }
        }
    }

    template <typename T, std::size_t Rows, std::size_t Columns>
    constexpr void MatrixImpl<T, Rows, Columns>::addition_tn_aux(const MatrixImpl &other, std::size_t start, std::size_t end) noexcept
    {
        /* For each row in A from Start to End of this chunk */
        for (std::size_t row{start}; row < end; row++)
        {
            for (std::size_t column{0}; column < columns(); column++)
            {
                /* update this->data_ for R values */
                data_[row][column] += other.data_[row][column];
            }
        }
    }

    /* single threaded (t1) implementation */
    template <typename T, std::size_t Rows, std::size_t Columns>
    constexpr MatrixImpl<T, Rows, Columns> MatrixImpl<T, Rows, Columns>::addition(const MatrixImpl &other) const & noexcept
    {
        MatrixImpl result{};
        addition_tn_aux(result, other, 0, rows());
        return result;
    }

    /* single threaded (t1) implementation for rvalues */
    template <typename T, std::size_t Rows, std::size_t Columns>
    constexpr MatrixImpl<T, Rows, Columns> MatrixImpl<T, Rows, Columns>::addition(const MatrixImpl &other) && noexcept
    {
        addition_tn_aux(other, 0, rows());
        return *this;
    }

    /* multi threaded (tn) implementation */
    template <typename T, std::size_t Rows, std::size_t Columns>
    MatrixImpl<T, Rows, Columns> MatrixImpl<T, Rows, Columns>::addition_tn(const MatrixImpl &other) const & noexcept
    {
        MatrixImpl result{};
        /* jthread is not supported in Clang 14 yet, so std::thread is being used instead. */
        std::vector<std::thread> threads{};
        for (auto &chunk : chunks_)
        {
            /* thread creation is costly, use thread pools for better performance */
            threads.emplace_back([this, &result, &other, &chunk]()
                                 { addition_tn_aux(result, other, chunk.first, chunk.second); });
        }

        for (auto &thread : threads)
        {
            thread.join();
        }

        return result;
    }

    /* multi threaded (tn) implementation for rvalues */
    template <typename T, std::size_t Rows, std::size_t Columns>
    MatrixImpl<T, Rows, Columns> MatrixImpl<T, Rows, Columns>::addition_tn(const MatrixImpl &other) && noexcept
    {
        /* jthread is not supported in Clang 14 yet, so std::thread is being used instead. */
        std::vector<std::thread> threads{};
        for (auto [start, end] : chunks_)
        {
            /* thread creation is costly, use thread pools for better performance */
            threads.emplace_back([this, &other, start, end]()
                                 { addition_tn_aux(other, start, end); });
        }

        for (auto &thread : threads)
        {
            thread.join();
        }

        return *this;
    }

    template <typename T, std::size_t Rows, std::size_t Columns>
    constexpr MatrixImpl<T, Rows, Columns> MatrixImpl<T, Rows, Columns>::operator-(const MatrixImpl &other) const & noexcept
    {
        MatrixImpl result{};
        for (std::size_t row{0}; row < rows(); row++)
        {
            for (std::size_t column{0}; column < columns(); column++)
            {
                result.data_[row][column] = data_[row][column] - other.data_[row][column];
            }
        }

        return result;
    }

    template <typename T, std::size_t Rows, std::size_t Columns>
    constexpr MatrixImpl<T, Rows, Columns> MatrixImpl<T, Rows, Columns>::operator-(const MatrixImpl &other) && noexcept
    {
        for (std::size_t row{0}; row < rows(); row++)
        {
            for (std::size_t column{0}; column < columns(); column++)
            {
                data_[row][column] -= other.data_[row][column];
            }
        }

        return *this;
    }
}
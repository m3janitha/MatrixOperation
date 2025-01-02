#pragma once

#include <array>
#include <vector>
#include <cstddef>
#include <iostream>
#include <thread>
#include <omp.h>

namespace matrix_tiled
{
    template <typename T, std::size_t Rows, std::size_t Columns>
    class MatrixImpl
    {
    public:
        static constexpr std::size_t BlockSize{4};
        using Block = std::array<std::array<T, BlockSize>, BlockSize>;
        using Data = std::array<std::array<Block, Columns / BlockSize>, Rows / BlockSize>;
        using ArrayData = std::array<std::array<T, Columns>, Rows>;

        constexpr MatrixImpl() = default;
        constexpr explicit MatrixImpl(const Data &data) : data_(data){};
        constexpr explicit MatrixImpl(Data &&data) : data_(std::move(data)){};
        /* conversion from 2D array */
        constexpr explicit MatrixImpl(const ArrayData &data);

        /* Public getters */
        static constexpr std::size_t rows() noexcept { return Rows; }
        static constexpr std::size_t columns() noexcept { return Columns; }
        static constexpr std::size_t row_blocks() noexcept { return Rows / BlockSize; }
        static constexpr std::size_t column_blocks() noexcept { return Columns / BlockSize; }

        /* Getter for data */
        [[nodiscard]] constexpr Data &data() { return data_; }
        [[nodiscard]] constexpr const Data &data() const { return data_; }
        [[nodiscard]] constexpr T &data_row_column(std::size_t row, std::size_t column) { return data_[row / BlockSize][column / BlockSize][row % BlockSize][column % BlockSize]; }
        [[nodiscard]] constexpr const T &data_row_column(std::size_t row, std::size_t column) const { return data_[row / BlockSize][column / BlockSize][row % BlockSize][column % BlockSize]; }

        template <std::size_t OtherColumns>
        constexpr MatrixImpl<T, Rows, OtherColumns> operator*(const MatrixImpl<T, Columns, OtherColumns> &other) const noexcept;

        /* Different multiplication implementations (public for user convenience) */
        template <std::size_t OtherColumns>
        constexpr MatrixImpl<T, Rows, OtherColumns> multiplication_naive(const MatrixImpl<T, Columns, OtherColumns> &other) const noexcept;

        // /* Cache optimised blocked (t1) implementation */
        // template <std::size_t OtherColumns>
        // constexpr MatrixImpl<T, Rows, OtherColumns> multiplication_blocked(const MatrixImpl<T, Columns, OtherColumns> &other) const noexcept;

        /* Cache optimised blocked (t1) implementation */
        template <std::size_t OtherColumns>
        constexpr MatrixImpl<T, Rows, OtherColumns> multiplication_tiled(const MatrixImpl<T, Columns, OtherColumns> &other) const noexcept;

        constexpr void multiplication_tiled_aux(const MatrixImpl<T, Rows, Columns>::Block &a, const MatrixImpl<T, Rows, Columns>::Block &b, MatrixImpl<T, Rows, Columns>::Block &r) const noexcept;

        auto operator<=>(const MatrixImpl &) const = default;

    private:
        /* Data */
        Data data_{};
    };

    template <typename T, std::size_t Rows, std::size_t Columns>
    constexpr MatrixImpl<T, Rows, Columns>::MatrixImpl(const ArrayData &data)
    {
        static_assert(Rows == Columns);
        static_assert(Rows % BlockSize == 0);

        for (std::size_t row{0}; row < rows(); row++)
        {
            for (std::size_t column{0}; column < columns(); column++)
            {
                data_[row / BlockSize][column / BlockSize][row % BlockSize][column % BlockSize] = data[row][column];
            }
        }
    }

    template <typename T, std::size_t Rows, std::size_t Columns>
    inline std::ostream &operator<<(std::ostream &os, const MatrixImpl<T, Rows, Columns> &matrix)
    {
        for (std::size_t row{0}; row < matrix.rows(); row++)
        {
            os << "[ ";
            for (std::size_t column{0}; column < matrix.columns(); column++)
            {
                os << matrix.data_row_column(row, column) << " ";
            }
            os << "]" << std::endl;
        }
        return os;
    }

    /* Configure this to select the optimal implementation based on matrix size */
    template <typename T, std::size_t Rows, std::size_t Columns>
    template <std::size_t OtherColumns>
    constexpr MatrixImpl<T, Rows, OtherColumns> MatrixImpl<T, Rows, Columns>::operator*(const MatrixImpl<T, Columns, OtherColumns> &other) const noexcept
    {
        return multiplication_naive(other);
    }

    /* Not cache friendly */
    template <typename T, std::size_t Rows, std::size_t Columns>
    template <std::size_t OtherColumns>
    constexpr MatrixImpl<T, Rows, OtherColumns> MatrixImpl<T, Rows, Columns>::multiplication_naive(const MatrixImpl<T, Columns, OtherColumns> &other) const noexcept
    {
        MatrixImpl<T, Rows, OtherColumns> result{};
        /* for each row in A */
        for (std::size_t row{0}; row < rows(); row++)
        {
            /* for each column in B */
            for (std::size_t other_col{0}; other_col < other.columns(); other_col++)
            {
                T sum{0};
                /* for each column in A / row in B */
                for (std::size_t column{0}; column < columns(); column++)
                {
                    sum += data_row_column(row, column) * other.data_row_column(column, other_col);
                }
                result.data_row_column(row, other_col) = sum;z
            }
        }

        return result;
    }

    // /* this doesn't work */
    // template <typename T, std::size_t Rows, std::size_t Columns>
    // template <std::size_t OtherColumns>
    // constexpr MatrixImpl<T, Rows, OtherColumns> MatrixImpl<T, Rows, Columns>::multiplication_blocked(const MatrixImpl<T, Columns, OtherColumns> &other) const noexcept
    // {
    //     static_assert(Rows == Columns);
    //     static_assert(Rows == OtherColumns);
    //     static_assert(Rows%BlockSize ==0);
    //     MatrixImpl<T, Rows, OtherColumns> result{};

    //     /* For each row in A */
    //     for (std::size_t i = 0; i < rows(); i++)
    //     {
    //         /* For each column in A (row in B), advance by Block Size */
    //         for (std::size_t i_block = 0; i_block < columns(); i_block += BlockSize)
    //         {
    //             // For each chunk of A/B for this block
    //             for (std::size_t k = 0; k < other.columns(); k += BlockSize)
    //             {
    //                 // For each row in the chunk
    //                 for (std::size_t k_block = 0; k_block < BlockSize; k_block++)
    //                 {
    //                     // Go through all the elements in the sub chunk
    //                     for (std::size_t idx = 0; idx < BlockSize; idx++)
    //                     {
    //                         result.data()[i / BlockSize][i_block][i % BlockSize][idx] += data()[i / BlockSize][k_block][i % BlockSize][k] * other.data()[k_block][i_block][k][idx];
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     return result;
    // }

    template <typename T, std::size_t Rows, std::size_t Columns>
    constexpr void MatrixImpl<T, Rows, Columns>::multiplication_tiled_aux(const MatrixImpl<T, Rows, Columns>::Block &a, const MatrixImpl<T, Rows, Columns>::Block &b, MatrixImpl<T, Rows, Columns>::Block &r) const noexcept
    {
        for (std::size_t x = 0; x < BlockSize; x++)
        {
            for (std::size_t y = 0; y < BlockSize; y++)
            {
                for (std::size_t z = 0; z < BlockSize; z++)
                {
                    r[x][z] += a[x][y] * b[y][z];
                }
            }
        }
    }

    template <typename T, std::size_t Rows, std::size_t Columns>
    template <std::size_t OtherColumns>
    constexpr MatrixImpl<T, Rows, OtherColumns> MatrixImpl<T, Rows, Columns>::multiplication_tiled(const MatrixImpl<T, Columns, OtherColumns> &other) const noexcept
    {
        static_assert(Rows == Columns);
        static_assert(Rows == OtherColumns);
        MatrixImpl<T, Rows, OtherColumns> result{};

        /* For each row block in A */
        for (std::size_t i = 0; i < row_blocks(); i++)
        {
            /* For each column block in A (row block in B) */
            for (std::size_t j = 0; j < column_blocks(); j++)
            {
                /* For each column block in B (column block in R) */
                for (std::size_t k{0}; k < other.column_blocks(); k++)
                {
                    // r[i][k] = a[i, j] * b[j][k]
                    auto &a = data()[i][j];
                    auto &b = other.data()[j][k];
                    auto &r = result.data()[i][k];
                    multiplication_tiled_aux(a, b, r);
                }
            }
        }

        return result;
    }

    template <std::size_t Rows, std::size_t Columns>
    using Matrix = MatrixImpl<double, Rows, Columns>;

}
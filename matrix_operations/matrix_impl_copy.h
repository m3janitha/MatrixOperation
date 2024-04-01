#pragma once

#include <array>
#include <vector>
#include <cstddef>
#include <iostream>
#include <thread>

namespace matrix
{
    template <template <class, std::size_t, std::size_t> class MultStrat, class T, std::size_t Rows, std::size_t Columns>
    class MatrixImpl
    {
    public:
        using Data = std::array<std::array<T, Columns>, Rows>;

        constexpr MatrixImpl() = default;
        constexpr explicit MatrixImpl(const Data &data) : data_(data){};

        /* public getters */
        static constexpr std::size_t rows() noexcept { return Rows; }
        static constexpr std::size_t columns() noexcept { return Columns; }

        constexpr Data &data() { return data_; }
        constexpr const Data &data() const { return data_; }

        /* Addition for lvlues */
        constexpr MatrixImpl
        operator+(const MatrixImpl &other) const & noexcept;
        /* Addition for rvalues (reuse the rvalue instead of allocatiing new) */
        constexpr MatrixImpl operator+(const MatrixImpl &other) && noexcept;

        // /* Subtraction for lvlues */
        // constexpr MatrixImpl operator-(const MatrixImpl &other) const & noexcept;
        // /* Subtraction for rvalues (reuse the rvalue instead of allocatiing new) */
        // constexpr MatrixImpl operator-(const MatrixImpl &other) && noexcept;

        // /* Scalar multiplication (matrix * scalar) */
        // constexpr MatrixImpl operator*(T scalar) const & noexcept;
        // /* Scalar multiplication (matrix * scalar) (reuse the rvalue instead of allocatiing new) */
        // constexpr MatrixImpl operator*(T scalar) && noexcept;

        // /* Scalar multiplication (scalar * matrix) */
        // friend constexpr MatrixImpl operator*(T scalar, const MatrixImpl &mat) noexcept { return mat * scalar; }
        // /* Scalar multiplication (scalar * matrix) */
        // friend constexpr MatrixImpl operator*(T scalar, MatrixImpl &&mat) noexcept { return std::move(mat) * scalar; }

        // template <std::size_t OtherColumns>
        // constexpr MatrixImpl<T, Rows, OtherColumns> operator*(const MatrixImpl<T, Columns, OtherColumns> &other) const noexcept;

        std::array<std::array<T, Columns>, Rows> data_{};
        static constexpr MultStrat<T, Rows, Columns> mult_strat_{};
    };

    template <template <class, std::size_t, std::size_t> class MultStrat, class T, std::size_t Rows, std::size_t Columns>
    inline std::ostream &operator<<(std::ostream &os, const MatrixImpl<MultStrat, T, Rows, Columns> &matrix)
    {
        for (std::size_t row{0}; row < matrix.rows(); row++)
        {
            os << "[ ";
            for (std::size_t column{0}; column < matrix.columns(); column++)
            {
                os << matrix.data_[row][column] << " ";
            }
            os << "]" << std::endl;
        }
        return os;
    }

    template <template <class, std::size_t, std::size_t> class MultStrat, class T, std::size_t Rows, std::size_t Columns>
    constexpr MatrixImpl<MultStrat, T, Rows, Columns> MatrixImpl<MultStrat, T, Rows, Columns>::operator+(const MatrixImpl &other) const & noexcept
    {
        MatrixImpl result{};
        for (std::size_t row{0}; row < rows(); row++)
        {
            for (std::size_t column{0}; column < columns(); column++)
            {
                result.data_[row][column] = data_[row][column] + other.data_[row][column];
            }
        }

        return result;
    }

    template <template <class, std::size_t, std::size_t> class MultStrat, class T, std::size_t Rows, std::size_t Columns>
    constexpr MatrixImpl<MultStrat, T, Rows, Columns> MatrixImpl<MultStrat, T, Rows, Columns>::operator+(const MatrixImpl &other) && noexcept
    {
        for (std::size_t row{0}; row < rows(); row++)
        {
            for (std::size_t column{0}; column < columns(); column++)
            {
                data_[row][column] += other.data_[row][column];
            }
        }

        return *this;
    }

    // template <template <class, std::size_t, std::size_t> class MultStrat, class T, std::size_t Rows, std::size_t Columns>
    // constexpr MatrixImpl<MultStrat, T, Rows, Columns> MatrixImpl<MultStrat, T, Rows, Columns>::operator-(const MatrixImpl &other) const & noexcept
    // {
    //     MatrixImpl result{};
    //     for (std::size_t row{0}; row < rows(); row++)
    //     {
    //         for (std::size_t column{0}; column < columns(); column++)
    //         {
    //             result.data_[row][column] = data_[row][column] - other.data_[row][column];
    //         }
    //     }

    //     return result;
    // }

    // template <template <class, std::size_t, std::size_t> class MultStrat, class T, std::size_t Rows, std::size_t Columns>
    // constexpr MatrixImpl<MultStrat, T, Rows, Columns> MatrixImpl<MultStrat, T, Rows, Columns>::operator-(const MatrixImpl &other) && noexcept
    // {
    //     for (std::size_t row{0}; row < rows(); row++)
    //     {
    //         for (std::size_t column{0}; column < columns(); column++)
    //         {
    //             data_[row][column] -= other.data_[row][column];
    //         }
    //     }

    //     return *this;
    // }

    // template <template <class, std::size_t, std::size_t> class MultStrat, class T, std::size_t Rows, std::size_t Columns>
    // constexpr MatrixImpl<MultStrat, T, Rows, Columns> MatrixImpl<MultStrat, T, Rows, Columns>::operator*(T scalar) const & noexcept
    // {
    //     MatrixImpl result{};
    //     for (std::size_t row{0}; row < rows(); row++)
    //     {
    //         for (std::size_t column{0}; column < columns(); column++)
    //         {
    //             result.data_[row][column] = data_[row][column] * scalar;
    //         }
    //     }
    //     return result;
    // }

    // template <template <class, std::size_t, std::size_t> class MultStrat, class T, std::size_t Rows, std::size_t Columns>
    // constexpr MatrixImpl<MultStrat, T, Rows, Columns> MatrixImpl<MultStrat, T, Rows, Columns>::operator*(T scalar) && noexcept
    // {
    //     std::cout << "rvalue optimised (matrix * scalar)" << std::endl;
    //     for (std::size_t row{0}; row < rows(); row++)
    //     {
    //         for (std::size_t column{0}; column < columns(); column++)
    //         {
    //             data_[row][column] *= scalar;
    //         }
    //     }
    //     return *this;
    // }

    // template <template <class, std::size_t, std::size_t> class MultStrat, class T, std::size_t Rows, std::size_t Columns>
    // template <std::size_t OtherColumns>
    // constexpr MatrixImpl<MultStrat, T, Rows, OtherColumns> MatrixImpl<MultStrat, T, Rows, Columns>::operator*(const MatrixImpl<MultStrat, class T, Columns, OtherColumns> &other) const noexcept
    // {
    //     return mult_strat_.multiplication(*this, other);
    // }
}
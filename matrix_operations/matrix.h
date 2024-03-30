#include <array>
#include <vector>
#include <cstddef>
#include <iostream>

template <std::size_t Rows, std::size_t Columns, typename T>
struct Matrix
{
    constexpr Matrix() = default;
    constexpr explicit Matrix(const std::array<std::array<T, Columns>, Rows> &data) : data_(data){};

    /* Addition for lvlues */
    constexpr Matrix operator+(const Matrix &other) const & noexcept;
    /* Addition for rvalues (reuse the rvalue instead of allocatiing new) */
    constexpr Matrix operator+(const Matrix &other) && noexcept;

    /* Subtraction for lvlues */
    constexpr Matrix operator-(const Matrix &other) const & noexcept;
    /* Subtraction for rvalues (reuse the rvalue instead of allocatiing new) */
    constexpr Matrix operator-(const Matrix &other) && noexcept;

    /* Scalar multiplication (matrix * scalar) */
    constexpr Matrix operator*(T scalar) const & noexcept
    {
        Matrix result{};
        for (std::size_t row{0}; row < rows(); row++)
        {
            for (std::size_t column{0}; column < columns(); column++)
            {
                result.data_[row][column] = data_[row][column] * scalar;
            }
        }
        return result;
    }

    /* Scalar multiplication (matrix * scalar) */
    constexpr Matrix operator*(T scalar) && noexcept
    {
        std::cout << "rvalue optimised (matrix * scalar)" << std::endl;
        for (std::size_t row{0}; row < rows(); row++)
        {
            for (std::size_t column{0}; column < columns(); column++)
            {
                data_[row][column] *= scalar;
            }
        }
        return *this;
    }

    /* Scalar multiplication (scalar * matrix) */
    friend constexpr Matrix operator*(T scalar, const Matrix &mat) noexcept
    {
        return mat * scalar;
    }

    /* Scalar multiplication (scalar * matrix) */
    friend constexpr Matrix operator*(T scalar, Matrix &&mat) noexcept
    {
        return mat * scalar;
    }

    template <std::size_t OtherColumns>
    constexpr Matrix<Rows, OtherColumns, T> operator*(const Matrix<Columns, OtherColumns, T> &other) const noexcept;

    constexpr std::size_t rows() const noexcept { return Rows; }
    constexpr std::size_t columns() const noexcept { return Columns; }

    /* The naive approach */
    template <std::size_t OtherColumns>
    constexpr Matrix<Rows, OtherColumns, T> multiplication_naive(const Matrix<Columns, OtherColumns, T> &other) const noexcept;

    /* Parallel approach */
    template <std::size_t OtherColumns>
    constexpr Matrix<Rows, OtherColumns, T> multiplication_parallel(const Matrix<Columns, OtherColumns, T> &other) const noexcept;
    template <std::size_t OtherColumns>
    constexpr Matrix<Rows, OtherColumns, T> multiplication_parallel_aux(Matrix<Columns, OtherColumns, T> &matt, const Matrix<Columns, OtherColumns, T> &other, std::size_t start, std::size_t end) const noexcept;

    inline static constexpr std::size_t number_of_worker_threads{8};
    std::array<std::array<T, Columns>, Rows> data_{};
};

// template <typename T>
// struct Matrix<1,1,T>
// {
//     constexpr Matrix() = default;
//     constexpr explicit Matrix(T data) : data_(data){};

//     constexpr Matrix<1, 1, T> operator+(const Matrix<1, 1, T> &other) const noexcept;

//     constexpr Matrix<Rows, Columns, T> operator-(const Matrix<Rows, Columns, T> &other) const noexcept;

//     template <std::size_t OtherColumns>
//     constexpr Matrix<Rows, OtherColumns, T> operator*(const Matrix<Columns, OtherColumns, T> &other) const noexcept;

//     constexpr std::size_t rows() const noexcept { return Rows; }
//     constexpr std::size_t columns() const noexcept { return Columns; }

//     T data_{};
// };

template <std::size_t Rows, std::size_t Columns, typename T>
inline std::ostream &operator<<(std::ostream &os, const Matrix<Rows, Columns, T> &matrix)
{
    for (std::size_t row{0}; row < matrix.rows(); row++)
    {
        os << "[ ";
        for (std::size_t column{0}; column < matrix.columns(); column++)
        {
            os << matrix.data_[row][column] << ", ";
        }
        os << "]" << std::endl;
    }
    return os;
}

template <std::size_t Rows, std::size_t Columns, typename T>
constexpr Matrix<Rows, Columns, T> Matrix<Rows, Columns, T>::operator+(const Matrix &other) const & noexcept
{
    Matrix result{};
    for (std::size_t row{0}; row < rows(); row++)
    {
        for (std::size_t column{0}; column < columns(); column++)
        {
            result.data_[row][column] = data_[row][column] + other.data_[row][column];
        }
    }

    return result;
}

template <std::size_t Rows, std::size_t Columns, typename T>
constexpr Matrix<Rows, Columns, T> Matrix<Rows, Columns, T>::operator+(const Matrix &other) && noexcept
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

template <std::size_t Rows, std::size_t Columns, typename T>
constexpr Matrix<Rows, Columns, T> Matrix<Rows, Columns, T>::operator-(const Matrix &other) const & noexcept
{
    Matrix result{};
    for (std::size_t row{0}; row < rows(); row++)
    {
        for (std::size_t column{0}; column < columns(); column++)
        {
            result.data_[row][column] = data_[row][column] - other.data_[row][column];
        }
    }

    return result;
}

template <std::size_t Rows, std::size_t Columns, typename T>
constexpr Matrix<Rows, Columns, T> Matrix<Rows, Columns, T>::operator-(const Matrix &other) && noexcept
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

// template <std::size_t Rows, std::size_t Columns, typename T>
// constexpr Matrix<Rows, Columns, T> Matrix<Rows, Columns, T>::operator*(T scalar) const & noexcept
// {
//     Matrix result{};
//     for (std::size_t row{0}; row < rows(); row++)
//     {
//         for (std::size_t column{0}; column < columns(); column++)
//         {
//             result.data_[row][column] = data_[row][column] * scalar;
//         }
//     }
//     return result;
// }

// template <std::size_t Rows, std::size_t Columns, typename T>
// constexpr Matrix<Rows, Columns, T> operator*(T scalar, const Matrix<Rows, Columns, T> &mat) noexcept
// {
//     Matrix result{};
//     return result;
// }

template <std::size_t Rows, std::size_t Columns, typename T>
template <std::size_t OtherColumns>
constexpr Matrix<Rows, OtherColumns, T> Matrix<Rows, Columns, T>::operator*(const Matrix<Columns, OtherColumns, T> &other) const noexcept
{
    return multiplication_naive(other);
}

/* The naive approach */
/* Not cache friendly */
template <std::size_t Rows, std::size_t Columns, typename T>
template <std::size_t OtherColumns>
constexpr Matrix<Rows, OtherColumns, T> Matrix<Rows, Columns, T>::multiplication_naive(const Matrix<Columns, OtherColumns, T> &other) const noexcept
{
    Matrix<Rows, OtherColumns, T> result{};
    for (std::size_t row{0}; row < rows(); row++)
    {
        for (std::size_t other_col{0}; other_col < other.columns(); other_col++)
        {
            T sum{0};
            for (std::size_t column{0}; column < columns(); column++)
            {
                sum += data_[row][column] * other.data_[column][other_col];
            }
            result.data_[row][other_col] = sum;
        }
    }

    return result;
}

template <std::size_t Rows, std::size_t Columns, typename T>
template <std::size_t OtherColumns>
constexpr Matrix<Rows, OtherColumns, T> Matrix<Rows, Columns, T>::multiplication_parallel_aux(Matrix<Columns, OtherColumns, T> &result, const Matrix<Columns, OtherColumns, T> &other, std::size_t start, std::size_t end) const noexcept
{
    for (std::size_t row{start}; row < end; row++)
    {
        for (std::size_t other_col{0}; other_col < other.columns(); other_col++)
        {
            T sum{0};
            for (std::size_t column{0}; column < columns(); column++)
            {
                sum += data_[row][column] * other.data_[column][other_col];
            }
            result.data_[row][other_col] = sum;
        }
    }
}

//     int n = vec.size();
//     int numSubVectors = n / m;
//     int remainingElements = n % m;

//     int startIndex = 0;
//     for (int i = 0; i < numSubVectors; ++i) {
//         std::vector<int> subVec(vec.begin() + startIndex, vec.begin() + startIndex + m);
//         result.push_back(subVec);
//         startIndex += m;
//     }

//     if (remainingElements > 0) {
//         std::vector<int> subVec(vec.begin() + startIndex, vec.begin() + startIndex + remainingElements);
//         result.push_back(subVec);
//     }

//     return result;
// }



template <std::size_t Rows, std::size_t Columns, typename T>
template <std::size_t OtherColumns>
constexpr Matrix<Rows, OtherColumns, T> Matrix<Rows, Columns, T>::multiplication_parallel(const Matrix<Columns, OtherColumns, T> &other) const noexcept
{


    Matrix<Rows, OtherColumns, T> result{};

    for (std::size_t row{0}; row < rows(); row++)
    {
        for (std::size_t other_col{0}; other_col < other.columns(); other_col++)
        {
            T sum{0};
            for (std::size_t column{0}; column < columns(); column++)
            {
                sum += data_[row][column] * other.data_[column][other_col];
            }
            result.data_[row][other_col] = sum;
        }
    }

    return result;
}
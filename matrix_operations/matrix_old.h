#include <array>
#include <vector>
#include <cstddef>
#include <iostream>
#include <thread>

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

    template <std::size_t OtherColumns>
    constexpr Matrix<Rows, OtherColumns, T> multiplication_naive_2(const Matrix<Columns, OtherColumns, T> &other) const noexcept;

    /* Parallel approach */
    template <std::size_t OtherColumns>
    constexpr Matrix<Rows, OtherColumns, T> multiplication_parallel(const Matrix<Columns, OtherColumns, T> &other) const noexcept;
    template <std::size_t OtherColumns>
    constexpr void multiplication_parallel_aux(Matrix<Rows, OtherColumns, T> &result, const Matrix<Columns, OtherColumns, T> &other, std::size_t start, std::size_t end) const noexcept;

    /* Blocked approach */
    template <std::size_t OtherColumns>
    constexpr Matrix<Rows, OtherColumns, T> multiplication_blocked(const Matrix<Columns, OtherColumns, T> &other) const noexcept;
    template <std::size_t OtherColumns>
    constexpr void multiplication_blocked_aux(Matrix<Rows, OtherColumns, T> &result, const Matrix<Columns, OtherColumns, T> &other, std::size_t start, std::size_t end) const noexcept;

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
constexpr Matrix<Rows, OtherColumns, T> Matrix<Rows, Columns, T>::multiplication_naive_2(const Matrix<Columns, OtherColumns, T> &other) const noexcept
{
    Matrix<Rows, OtherColumns, T> result{};
    /* rows of first */
    for (std::size_t i{0}; i < rows(); i++)
    {
        /* columns of first */
        for (std::size_t k{0}; k < columns(); k++)
        {
            //auto aik = data_[i * rows() + k];
            auto data_ik = data_[i][k];
            /* columns of second */
            for (std::size_t j{0}; j < other.columns(); j++)
            {
                //result.data_[i * rows() + j] += aik * other.data_[k * other.columns() + j];
                result.data_[i][j] += data_ik * other.data_[k][j];
            }
        }
    }

    return result;
}

inline static auto get_parallel_blocks(const std::size_t array_length, const std::size_t number_of_worker_threads)
{
    std::size_t number_of_blocks = number_of_worker_threads;
    std::size_t block_size = array_length / number_of_blocks;
    std::size_t remainder  = array_length % number_of_worker_threads;    

    std::vector<std::pair<std::size_t, std::size_t>> blocks{};
    std::size_t start_index{0};
    for (std::size_t i{0}; i < number_of_blocks; i++)
    {
        auto block_length = block_size + (i < remainder? 1:0);
        blocks.emplace_back(start_index, start_index + block_length);
        start_index += block_length;
    }
    return blocks;
}

template <std::size_t Rows, std::size_t Columns, typename T>
template <std::size_t OtherColumns>
constexpr void Matrix<Rows, Columns, T>::multiplication_parallel_aux(Matrix<Rows, OtherColumns, T> &result, const Matrix<Columns, OtherColumns, T> &other, std::size_t start, std::size_t end) const noexcept
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

template <std::size_t Rows, std::size_t Columns, typename T>
template <std::size_t OtherColumns>
constexpr Matrix<Rows, OtherColumns, T> Matrix<Rows, Columns, T>::multiplication_parallel(const Matrix<Columns, OtherColumns, T> &other) const noexcept
{
    auto blocks = get_parallel_blocks(rows(), number_of_worker_threads);
    // std::cout << "threads: " << number_of_worker_threads << std::endl;
    // std::cout << "blocks: ";
    // for (auto b : blocks)
    //     std::cout << b.first << ":" << b.second << " ";
    Matrix<Rows, OtherColumns, T> result{};
    std::vector<std::thread> threads{};
    for (auto &block : blocks)
    {
        /* thread creation is costly, use thread pools for better performance */
        threads.emplace_back([this, &result, &other, &block]()
                             { multiplication_parallel_aux(result, other, block.first, block.second); });
    }

    for (auto &thread : threads)
    {
        thread.join();
    }

    return result;
}

template <std::size_t Rows, std::size_t Columns, typename T>
template <std::size_t OtherColumns>
constexpr void Matrix<Rows, Columns, T>::multiplication_blocked_aux(Matrix<Rows, OtherColumns, T> &result, const Matrix<Columns, OtherColumns, T> &other, std::size_t start, std::size_t end) const noexcept
{
    /* A * B = C*/
    /* each row of */
    // for (std::size_t row{start}; row < end; row++)
    // {
    //     // For each block in the row...
    //     // Solve for 16 elements at a time
    //     for (std::size_t block = 0; block < N; block += 16)
    //     {
    //         // For each chunk of A/B for this block
    //         for (std::size_t chunk = 0; chunk < N; chunk += 16)
    //         {
    //              // For each row in the chunk
    //             for (std::size_t sub_chunk = 0; sub_chunk < 16; sub_chunk++){
    //                 // Go through all the elements in the sub chunk
    //                 for (std::size_t idx = 0; idx < 16; idx++){
    //                     // C[row * N + block + idx] ->result[row][block+idx]
    //                     // A[row * N + chunk + sub_chunk] -> this[row][chunk+sub]
    //                     // B[chunk * N + sub_chunk * N + block + idx]; -> other[]
    //                     result[row * N + block + idx] = A[row * N + chunk + sub_chunk] *
    //                         B[chunk * N + sub_chunk * N + block + idx];
    //                 }
    //             }
    //         }
    //     }
    // }
}

template <std::size_t Rows, std::size_t Columns, typename T>
template <std::size_t OtherColumns>
constexpr Matrix<Rows, OtherColumns, T> Matrix<Rows, Columns, T>::multiplication_blocked(const Matrix<Columns, OtherColumns, T> &other) const noexcept
{
    auto blocks = get_parallel_blocks(rows(), number_of_worker_threads);
    Matrix<Rows, OtherColumns, T> result{};
    std::vector<std::thread> threads{blocks.size()};
    for (auto &block : blocks)
    {
        /* thread creation is costly, use thread pools for better performance */
        threads.emplace_back([this, &result, &other, &block]()
                             { multiplication_blocked_aux(result, other, block.first, block.second); });
    }

    for (auto &thread : threads)
    {
        thread.join();
    }

    return result;
}
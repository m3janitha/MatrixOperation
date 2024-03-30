// template <std::size_t Rows, std::size_t Columns, typename T>
// inline std::ostream &operator<<(std::ostream &os, const Matrix<Rows, Columns, T> &matrix)
// {
//     for (std::size_t row{0}; row < matrix.rows(); row++)
//     {
//         os << "[ ";
//         for (std::size_t column{0}; column < matrix.columns(); column++)
//         {
//             os << matrix.data_[row][column] << ", ";
//         }
//         os << "]" << std::endl;
//     }
//     return os;
// }

// template <std::size_t Rows, std::size_t Columns, typename T>
// constexpr Matrix<Rows, Columns, T> Matrix<Rows, Columns, T>::operator+(const Matrix<Rows, Columns, T> &other) const & noexcept
// {
//     Matrix<Rows, Columns, T> sum{};
//     for (std::size_t row{0}; row < rows(); row++)
//     {
//         for (std::size_t column{0}; column < columns(); column++)
//         {
//             sum.data_[row][column] = data_[row][column] + other.data_[row][column];
//         }
//     }

//     return sum;
// }

// template <std::size_t Rows, std::size_t Columns, typename T>
// constexpr Matrix<Rows, Columns, T> Matrix<Rows, Columns, T>::operator+(const Matrix<Rows, Columns, T> &other) && noexcept
// {
//     for (std::size_t row{0}; row < rows(); row++)
//     {
//         for (std::size_t column{0}; column < columns(); column++)
//         {
//             data_[row][column] += other.data_[row][column];
//         }
//     }

//     return *this;
// }

// template <std::size_t Rows, std::size_t Columns, typename T>
// constexpr Matrix<Rows, Columns, T> Matrix<Rows, Columns, T>::operator-(const Matrix<Rows, Columns, T> &other) const & noexcept
// {
//     Matrix<Rows, Columns, T> sum{};
//     for (std::size_t row{0}; row < rows(); row++)
//     {
//         for (std::size_t column{0}; column < columns(); column++)
//         {
//             sum.data_[row][column] = data_[row][column] - other.data_[row][column];
//         }
//     }

//     return sum;
// }

// template <std::size_t Rows, std::size_t Columns, typename T>
// constexpr Matrix<Rows, Columns, T> Matrix<Rows, Columns, T>::operator-(const Matrix<Rows, Columns, T> &other) && noexcept
// {
//     for (std::size_t row{0}; row < rows(); row++)
//     {
//         for (std::size_t column{0}; column < columns(); column++)
//         {
//             data_[row][column] -= other.data_[row][column];
//         }
//     }

//     return sum;
// }

// template <std::size_t Rows, std::size_t Columns, typename T>
// template <std::size_t OtherColumns>
// constexpr Matrix<Rows, OtherColumns, T> Matrix<Rows, Columns, T>::operator*(const Matrix<Columns, OtherColumns, T> &other) const noexcept
// {
//     Matrix<Rows, OtherColumns, T> mul{};
//     for (std::size_t row{0}; row < rows(); row++)
//     {
//         for (std::size_t other_col{0}; other_col < other.columns(); other_col++)
//         {
//             T sum{0};
//             for (std::size_t column{0}; column < columns(); column++)
//             {
//                 sum += data_[row][column] * other.data_[column][other_col];
//             }
//             mul.data_[row][other_col] = sum;
//         }
//     }

//     return mul;
// }
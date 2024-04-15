#pragma once

#include <vector>
#include <array>
#include <iostream>
#include <matrix_operations/matrix.h>

inline std::ostream &operator<<(std::ostream &os, const std::vector<std::vector<double>> &matrix)
{
    for (std::size_t row{0}; row < matrix.size(); row++)
    {
        os << "[ ";
        for (std::size_t column{0}; column < matrix.size(); column++)
        {
            os << matrix[row][column] << " ";
        }
        os << "]" << std::endl;
    }
    return os;
}

namespace strassens
{

    inline std::vector<std::vector<double>> mat_add(const std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b)
    {
        auto N = a.size();
        std::vector<std::vector<double>> r(N, std::vector<double>(N, 0));
        for (std::size_t i = 0; i < N; i++)
        {
            for (std::size_t j = 0; j < N; j++)
            {
                r[i][j] = a[i][j] + b[i][j];
            }
        }

        return r;
    }

    inline std::vector<std::vector<double>> mat_sub(const std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b)
    {
        auto N = a.size();
        std::vector<std::vector<double>> r(N, std::vector<double>(N, 0));
        for (std::size_t i = 0; i < N; i++)
        {
            for (std::size_t j = 0; j < N; j++)
            {
                r[i][j] = a[i][j] - b[i][j];
            }
        }

        return r;
    }

    inline std::vector<std::vector<double>> mat_mul(const std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b)
    {
        auto N = a.size();

        std::vector<std::vector<double>> r(N, std::vector<double>(N, 0));
        // base case
        if (N == 1)
        {
            r[0][0] = a[0][0] * b[0][0];
            return r;
        }

        auto mid = N / 2;

        /* initilize vectors to size with value 0 */
        std::vector<std::vector<double>> a11(mid, std::vector<double>(mid, 0));
        std::vector<std::vector<double>> a12(mid, std::vector<double>(mid, 0));
        std::vector<std::vector<double>> a21(mid, std::vector<double>(mid, 0));
        std::vector<std::vector<double>> a22(mid, std::vector<double>(mid, 0));

        std::vector<std::vector<double>> b11(mid, std::vector<double>(mid, 0));
        std::vector<std::vector<double>> b12(mid, std::vector<double>(mid, 0));
        std::vector<std::vector<double>> b21(mid, std::vector<double>(mid, 0));
        std::vector<std::vector<double>> b22(mid, std::vector<double>(mid, 0));

        /* Fill A & B */
        for (std::size_t i = 0; i < mid; i++)
        {
            for (std::size_t j = 0; j < mid; j++)
            {
                /* each block is size of mid, so mid + i & mid + j */
                a11[i][j] = a[i][j];
                a12[i][j] = a[i][mid + j];
                a21[i][j] = a[mid + i][j];
                a22[i][j] = a[mid + i][mid + j];

                b11[i][j] = b[i][j];
                b12[i][j] = b[i][mid + j];
                b21[i][j] = b[mid + i][j];
                b22[i][j] = b[mid + i][mid + j];
            }
        }

        // P1 = A11 * (B12 - B22)
        // P2 = (A11 + A12) * B22
        // P3 = (A21 + A22) * B11
        // P4 = A22 * (B21 - B11)
        // P5 = (A11 + A22) * (B11 + B22)
        // P6 = (A12 - A22) * (B21 + B22)
        // P7 = (A11 - A21) * (B11 + B12)

        auto p1 = mat_mul(a11, mat_sub(b12, b22));
        auto p2 = mat_mul(mat_add(a11, a12), b22);
        auto p3 = mat_mul(mat_add(a21, a22), b11);
        auto p4 = mat_mul(a22, mat_sub(b21, b11));
        auto p5 = mat_mul(mat_add(a11, a22), mat_add(b11, b22));
        auto p6 = mat_mul(mat_sub(a12, a22), mat_add(b21, b22));
        auto p7 = mat_mul(mat_sub(a11, a21), mat_add(b11, b12));

        // C11 = P5 + P4 - P2 + P6
        // C12 = P1 + P2
        // C21 = P3 + P4
        // C22 = P5 + P1 - P3 - P7

        auto c11 = mat_sub(mat_add(mat_add(p5, p4), p6), p2);
        auto c12 = mat_add(p1, p2);
        auto c21 = mat_add(p3, p4);
        auto c22 = mat_sub(mat_sub(mat_add(p5, p1), p3), p7);

        for (std::size_t i = 0; i < mid; i++)
        {
            for (std::size_t j = 0; j < mid; j++)
            {
                r[i][j] = c11[i][j];
                r[i][mid + j] = c12[i][j];
                r[mid + i][j] = c21[i][j];
                r[mid + i][mid + j] = c22[i][j];
            }
        }

        return r;
    }

    inline bool is_power_of_two(std::size_t n)
    {
        // Check if N is non-negative and has only one bit set
        return n > 0 && (n & (n - 1)) == 0;
    }

    inline std::vector<std::vector<double>> strassens_mult(const std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b)
    {
        // validate preconditions
        if ((a.size() == a[0].size()) && (a.size() == b.size()) && (b.size() == b[0].size()))
        {
            if (is_power_of_two(a.size()))
            {
                return mat_mul(a, b);
            }
        }

        std::cout << "Invalid Input" << std::endl;
        return {};
    }

    template <typename T>
    inline std::vector<std::vector<double>> array_to_vec_matrix(const T &mat)
    {
        std::vector<std::vector<double>> r(mat.rows(), std::vector<double>(mat.columns(), 0));
        for (std::size_t row{0}; row < mat.rows(); row++)
        {
            for (std::size_t column{0}; column < mat.columns(); column++)
            {
                r[row][column] = mat.data()[row][column];
            }
        }
        return r;
    }

    template <std::size_t Columns, std::size_t Rows>
    inline std::array<std::array<double, Columns>, Rows> vec_matrix_to_array(const std::vector<std::vector<double>> &mat)
    {
        std::array<std::array<double, Columns>, Rows> r{};
        for (std::size_t row{0}; row < mat.size(); row++)
        {
            for (std::size_t column{0}; column < mat[0].size(); column++)
            {
                r[row][column] = mat[row][column];
            }
        }
        return r;
    }
}
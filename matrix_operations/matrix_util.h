#pragma once

#include <matrix_operations/matrix.h>
#include <random>
#include <ctime>
#include <iostream>

using namespace matrix;

template <typename T, typename M>
inline void fill_matrix(M &m)
{
    /* Create random number generator */
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<double> dist(-10, 10);

    for (std::size_t r = 0; r < m.rows(); r++)
    {
        for (std::size_t c = 0; c < m.columns(); c++)
        {
            m.data()[r][c] = static_cast<T>(dist(rng));
        }
    }
}

template <typename T, typename M>
inline void fill_matrix2(M &m)
{
    /* Create random number generator */
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<T> dist(-10, 10);

    for (std::size_t r = 0; r < m.rows(); r++)
    {
        for (std::size_t c = 0; c < m.columns(); c++)
        {
            m.data_row_column(r,c) = dist(rng);
        }
    }
}
#include "matrix.h"
#include <iostream>
#include <vector>

auto get_parallel_blocks(const std::size_t array_length, const std::size_t block_size)
{
    std::size_t number_of_blocks = array_length / block_size;
    std::size_t remaining = array_length % block_size;

    std::vector<std::pair<std::size_t,std::size_t>> blocks{};
    std::size_t start_index{0};
    for(std::size_t i{0}; i<number_of_blocks; i++){
        blocks.emplace_back(start_index, start_index+block_size-1);
        start_index+=block_size;
    }
    if(remaining >0){
        blocks.emplace_back(start_index, start_index+remaining-1);
    }
    return blocks;
}

int main()
{
    constexpr std::array<std::array<int, 3>, 2> a1{{{1, 2, 3}, {4, 5, 6}}};
    constexpr Matrix<2, 3, int> m1(a1);
    constexpr Matrix m2 = m1;
    constexpr auto m3 = m1 + m2;
    constexpr auto m4 = m3 - m1;

    constexpr std::array<std::array<int, 4>, 3> ax{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 1, 2, 3}}};
    constexpr Matrix<3, 4, int> mx(ax);

    constexpr auto mxx = m1 * mx;

    std::cout << "matrix 1" << std::endl
              << m1 << std::endl;
    std::cout << "matrix x" << std::endl
              << mx << std::endl;
    std::cout << "matrix xx" << std::endl
              << mxx << std::endl;

    constexpr auto maa = m1 * 2;
    std::cout << "scalar m1 * 2" << std::endl
              << maa << std::endl;
    std::cout << "scalar m1 * 2 * 3" << std::endl
              << m1 * 2 * 3 << std::endl;
    std::cout << "matrix 1" << std::endl
              << m1 << std::endl;
    constexpr auto mbb = 2 * m1;
    std::cout << "scalar 2 * m1" << std::endl
              << mbb << std::endl;

    auto testv = get_parallel_blocks(10,10);
    for(auto& a:testv)
    {
        std::cout << a.first << "," << a.second << "  ";
    }
    // std::cout << "matrix 2" << std::endl
    //           << m2 << std::endl;
    // std::cout << "matrix 3" << std::endl
    //           << m3 << std::endl;
    // std::cout << "matrix 4" << std::endl
    //           << m4 << std::endl;
    // std::cout << "matrix 3" << std::endl
    //           << m3 << std::endl;
    return testv[0].first;
}
#include <matrix_operations/matrix.h>
#include <matrix_operations/solution.h>
#include <iostream>
#include <vector>

using namespace matrix;

int main()
{
    constexpr Matrix<1, 3> aa1{{{{1, 2, 3}}}};
    constexpr Matrix<3, 1> aa2{{{{1}, {2}, {3}}}};
    auto aa3 = aa1 * aa2;
    std::cout << "aa3 = aa1*aa2" << std::endl
              << aa3 << std::endl;

    constexpr Matrix<2, 3> m1{{{{1, 2, 3}, {4, 5, 6}}}};
    std::cout << "matrix 1" << std::endl
              << m1 << std::endl;

    constexpr auto m2 = m1;
    constexpr Matrix<3, 4> mx{{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 1, 2, 3}}}};
    std::cout << "matrix x" << std::endl
              << mx << std::endl;

    /*constexpr*/ auto mxx = m1 * mx;
    std::cout << "matrix  m1 * mx" << std::endl
              << mxx << std::endl;

    // std::cout << "matrix blocked m1 * mx" << std::endl
    //           << m1.multiplication_blocked(mx) << std::endl;

    constexpr auto maa = m1 * 2.0;
    std::cout << "scalar m1 * 2" << std::endl
              << maa << std::endl;

    std::cout << "scalar m1 * 2 * 3" << std::endl
              << m1 * 2 * 3 << std::endl;

    constexpr auto mxx2 = m1.multiplication_t1(mx);
    std::cout << "multiplication t1 xx" << std::endl
              << mxx2 << std::endl;

    auto mtnxx2 = m1.multiplication_tn(mx);
    std::cout << "multiplication tn xx" << std::endl
              << mtnxx2 << std::endl;

    Matrix<4, 4> mx1{{{{1, 2, 0, 1}, {2, 1, 0, 2}, {0, 2, 1, 1}, {2, 1, 0, 2}}}};
    Matrix<4, 4> mx2{{{{1, 1, 0, 2}, {2, 1, 1, 2}, {1, 2, 1, 0}, {0, 2, 1, 1}}}};

    Matrix<4, 4> mx3{{{{2, 1, 0, 1}, {0, 2, 1, 0}, {1, 0, 2, 1}, {2, 1, 1, 2}}}};

    std::cout << "multiplication t1 mx1 x mx2 " << std::endl
              << mx1.multiplication_t1(mx2) << std::endl;

    std::cout << "multiplication blocked mx1 x mx2 " << std::endl
              << mx1.multiplication_blocked(mx2) << std::endl;

    if(mx1.multiplication_t1(mx2) != mx1.multiplication_blocked(mx2)){
        std::cout << "==================== fail =================" << std::endl;
    }else{
        std::cout << "go" << std::endl;
    }

    constexpr auto m3 = m1 + m2;
    constexpr auto m4 = m3 - m1;

    std::cout << "m1 + m2" << std::endl
              << m3 << std::endl;

    std::cout << "m3 - m1" << std::endl
              << m4 << std::endl;

    std::cout << "m1 + m2 + m3" << std::endl
              << (m1 + m2) + m3 << std::endl;

    std::cout << "m1 + m2 tn" << std::endl
              << m1.addition_tn(m2) << std::endl;

    std::cout << "m1 + m2 + m3 tn" << std::endl
              << m1.addition_tn(m2).addition_tn(m3) << std::endl;

    auto xxx = ab_c_generic(mx1, mx2, mx3);
    std::cout << "ab_c_generic" << std::endl
              << xxx << std::endl;

    std::cout << "ab_c_optimised" << std::endl
              << ab_c_optimised(mx1, mx2, mx3) << std::endl;

    std::cout << "ab_c_optimised_tn" << std::endl
              << ab_c_optimised_tn(mx1, mx2, mx3) << std::endl;

    if (m1 == m2)
    {
        std::cout << "m1 == m2" << std::endl;
    }

    if (mx1 == mx2)
    {
        std::cout << "mx1 == mx2" << std::endl;
    }
    else
    {
        std::cout << "mx1 != mx2" << std::endl;
    }

    // std::cout << "mx1 - mx2" << std::endl
    //           << mx1 - mx2 << std::endl;

    // auto xxx = mx1 * mx2 + mx3;
    // std::cout << "mx1 * mx2 + mx3" << std::endl
    //           << xxx << std::endl;

    // auto yyy = ab_c(mx1, mx2, mx3);
    // std::cout << "mx1 * mx2 + mx3 abc" << std::endl
    //     << yyy << std::endl;

    return 0;
}
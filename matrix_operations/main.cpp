#include <matrix_operations/matrix.h>
#include <matrix_operations/solution.h>
#include <iostream>
#include <vector>

using namespace matrix;

int main()
{
    {
        std::cout << "=================== Demo ===================" << std::endl;
        constexpr Matrix<1, 3> a{{{{1, 2, 3}}}};
        constexpr Matrix<3, 1> b{{{{1}, {2}, {3}}}};

        constexpr auto r = a * b;
        std::cout << "====== A ======" << std::endl
                  << a << std::endl
                  << "====== B ======" << std::endl
                  << b << std::endl
                  << "==== A x B ====" << std::endl
                  << r << std::endl;
    }
    {
        std::cout << "=================== Demo ===================" << std::endl;
        constexpr Matrix<2, 3> a{{{{1, 2, 3}, {4, 5, 6}}}};
        constexpr Matrix<3, 4> b{{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 1, 2, 3}}}};

        constexpr auto r = a * b;
        std::cout << "====== A ======" << std::endl
                  << a << std::endl
                  << "====== B ======" << std::endl
                  << b << std::endl
                  << "==== A x B ====" << std::endl
                  << r << std::endl;
    }

    {
        std::cout << "=================== Demo ===================" << std::endl;
        constexpr Matrix<2, 3> a{{{{1, 2, 3}, {4, 5, 6}}}};
        constexpr double b{2.0};

        constexpr auto r = a * b;
        std::cout << "====== A ======" << std::endl
                  << a << std::endl
                  << "====== B ======" << std::endl
                  << b << std::endl
                  << "==== A x B ====" << std::endl
                  << r << std::endl;
    }

    {
        std::cout << "=================== Demo ===================" << std::endl;
        constexpr Matrix<2, 3> a{{{{1, 2, 3}, {4, 5, 6}}}};
        constexpr double b{2.0};
        constexpr double c{2.0};

        constexpr auto r = a * b * c;
        std::cout << "====== A ======" << std::endl
                  << a << std::endl
                  << "====== B ======" << std::endl
                  << b << std::endl
                  << "====== c ======" << std::endl
                  << c << std::endl
                  << "== A x B x C ==" << std::endl
                  << r << std::endl;
    }

    {
        std::cout << "=================== Demo ===================" << std::endl;
        constexpr Matrix<4, 4> a{{{{1, 2, 0, 1}, {2, 1, 0, 2}, {0, 2, 1, 1}, {2, 1, 0, 2}}}};
        constexpr Matrix<4, 4> b{{{{1, 1, 0, 2}, {2, 1, 1, 2}, {1, 2, 1, 0}, {0, 2, 1, 1}}}};

        constexpr auto r = a + b;
        std::cout << "====== A ======" << std::endl
                  << a << std::endl
                  << "====== B ======" << std::endl
                  << b << std::endl
                  << "==== A + B ====" << std::endl
                  << r << std::endl;
    }

    {
        std::cout << "=================== Demo ===================" << std::endl;
        constexpr Matrix<4, 4> a{{{{1, 2, 0, 1}, {2, 1, 0, 2}, {0, 2, 1, 1}, {2, 1, 0, 2}}}};
        constexpr Matrix<4, 4> b{{{{1, 1, 0, 2}, {2, 1, 1, 2}, {1, 2, 1, 0}, {0, 2, 1, 1}}}};

        constexpr auto r = a - b;
        std::cout << "====== A ======" << std::endl
                  << a << std::endl
                  << "====== B ======" << std::endl
                  << b << std::endl
                  << "==== A - B ====" << std::endl
                  << r << std::endl;
    }

    return 0;
}
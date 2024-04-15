#include <matrix_operations/matrix.h>
#include <matrix_operations/solution.h>
#include <iostream>
#include <vector>
#include <matrix_operations/strassens_algorithm.h>
#include <matrix_operations/matrix_impl_2.h>
#include <matrix_operations/matrix_util.h>

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

    {
        std::cout << "=================== Demo ===================" << std::endl;
        constexpr Matrix<4, 4> a{{{{1, 2, 0, 1}, {2, 1, 0, 2}, {0, 2, 1, 1}, {2, 1, 0, 2}}}};
        constexpr Matrix<4, 4> b{{{{1, 1, 0, 2}, {2, 1, 1, 2}, {1, 2, 1, 0}, {0, 2, 1, 1}}}};

        constexpr auto r = a * b;
        auto aa = strassens::array_to_vec_matrix(a);
        auto bb = strassens::array_to_vec_matrix(b);

        std::cout << "====== A ======" << std::endl
                  << a << std::endl
                  << "====== B ======" << std::endl
                  << b << std::endl
                  << "==== A x B ====" << std::endl
                  << r << std::endl;
    }

    {
        std::cout << "=================== Demo Blocked ===================" << std::endl;
        Matrix<8, 8> a;
        Matrix<8, 8> b;

        fill_matrix<int>(a);
        fill_matrix<int>(b);

        matrix_tiled::Matrix<8, 8> a2{a.data()};
        matrix_tiled::Matrix<8, 8> b2{b.data()};

        std::cout << "====== A ======" << std::endl
                  << a << std::endl
                  << "====== B ======" << std::endl
                  << b << std::endl
                  << "==== A x B ====" << std::endl
                  << a * b << std::endl
                  << "==== A x B tiled ====" << std::endl
                  << a2.multiplication_tiled(b2) << std::endl;

        matrix_tiled::Matrix<8, 8> r2{(a * b).data()};
        if ((a2 * b2) == r2)
        {
            std::cout << " A * B == A * B Blocked" << std::endl;
        }
    }

    {
        std::cout << "=================== Demo strassens ===================" << std::endl;
        constexpr Matrix<4, 4> a{{{{1, 2, 0, 1}, {2, 1, 0, 2}, {0, 2, 1, 1}, {2, 1, 0, 2}}}};
        constexpr Matrix<4, 4> b{{{{1, 1, 0, 2}, {2, 1, 1, 2}, {1, 2, 1, 0}, {0, 2, 1, 1}}}};

        auto aa = strassens::array_to_vec_matrix(a);
        auto bb = strassens::array_to_vec_matrix(b);

        std::cout << "====== A ======" << std::endl
                  << a << std::endl
                  << "====== B ======" << std::endl
                  << b << std::endl
                  << "==== A x B strassens ====" << std::endl
                  << strassens::strassens_mult(aa, bb) << std::endl;
    }

    {
        std::cout << "=================== Demo ===================" << std::endl;
        constexpr Matrix<4, 4> a{{{{1, 2, 0, 1}, {2, 1, 0, 2}, {0, 2, 1, 1}, {2, 1, 0, 2}}}};
        constexpr Matrix<4, 4> b{{{{1, 1, 0, 2}, {2, 1, 1, 2}, {1, 2, 1, 0}, {0, 2, 1, 1}}}};

        constexpr auto r = a * 2;
        std::cout << "====== A ======" << std::endl
                  << a << std::endl
                  << "==== A * 2 ====" << std::endl
                  << r << std::endl
                  << "==== 2 * A ====" << std::endl
                  << 2 * a << std::endl;
    }

    {
        std::cout << "=================== Demo Thread Pool ===================" << std::endl;
        Matrix<8, 8> a{};
        Matrix<8, 8> b{};
        fill_matrix<int>(a);
        fill_matrix<int>(b);

        static thread_pool::ThreadPoolInstance tpi;

        auto r = a.multiplication_tn_pool(b);
        std::cout << "====== A ======" << std::endl
                  << a << std::endl
                  << "====== B ======" << std::endl
                  << b << std::endl
                  << "==== A * B ====" << std::endl
                  << r << std::endl;
    }

    return 0;
}
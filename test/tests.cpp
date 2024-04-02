#include <gtest/gtest.h>
#include <matrix_operations/matrix.h>
#include <matrix_operations/solution.h>
#include <matrix_operations/matrix_util.h>

using namespace std::string_literals;
using namespace ::matrix;

/* Addition: A + B = R */
template <typename A, typename B, typename R>
void validate_addition(A &a, B &b, R &r)
{
    EXPECT_EQ(a + b, r);
    EXPECT_EQ(a.addition(b), r);
    EXPECT_EQ(a.addition_tn(b), r);
}

/* 1 X 2 */
TEST(Addition, 1_by_2_matrix)
{
    Matrix<1, 2> a{{{2, 3}}};
    Matrix<1, 2> b{{{1, 4}}};
    Matrix<1, 2> r{{{3, 7}}};

    validate_addition(a, b, r);
}

/* 2 X 1 */
TEST(Addition, 2_by_1_matrix)
{

    Matrix<2, 1> a{{{{1}, {5}}}};
    Matrix<2, 1> b{{{{3}, {4}}}};
    Matrix<2, 1> r{{{{4}, {9}}}};

    validate_addition(a, b, r);
}

/* 3 x 3 */
TEST(Addition, 3_by_3_matrix)
{

    Matrix<3, 3> a{{{{1, 2, 3}, {2, 1, 3}, {3, 2, 1}}}};
    Matrix<3, 3> b{{{{1, 1, 1}, {3, 1, 1}, {4, 2, 5}}}};
    Matrix<3, 3> r{{{{2, 3, 4}, {5, 2, 4}, {7, 4, 6}}}};

    validate_addition(a, b, r);
}

/* Subtraction: A - B = R */
/* Not required for this solution, just for clarity */
TEST(Subtraction, 3_by_3_matrix)
{
    Matrix<3, 3> a{{{{1, 2, 3}, {2, 1, 3}, {3, 2, 1}}}};
    Matrix<3, 3> b{{{{1, 1, 1}, {3, 1, 1}, {4, 2, 5}}}};
    Matrix<3, 3> r{{{{0, 1, 2}, {-1, 0, 2}, {-1, 0, -4}}}};

    EXPECT_EQ(a - b, r);
}

/* Multiplication: A . B = R */
template <typename A, typename B, typename R>
void validate_multiplication(A &a, B &b, R &r)
{
    EXPECT_EQ(a * b, r);
    EXPECT_EQ(a.multiplication_naive(b), r);
    EXPECT_EQ(a.multiplication_t1(b), r);
    EXPECT_EQ(a.multiplication_tn(b), r);
    EXPECT_EQ(a.multiplication_omp(b), r);
}

/* A: 1 X 2  B:2 X 1 R: 1 * 1 */
TEST(Multiplication, 1x2_2X1_matrices)
{
    Matrix<1, 2> a{{{{1, 2}}}};
    Matrix<2, 1> b{{{{3}, {4}}}};
    Matrix<1, 1> r{{{{11}}}};
    validate_multiplication(a, b, r);
}

/* A: 2 X 1  B:1 X 2 R: 2 * 2 */
TEST(Multiplication, 2x1_1X2_matrices)
{
    Matrix<2, 1> a{{{{3}, {4}}}};
    Matrix<1, 2> b{{{{1, 2}}}};
    Matrix<2, 2> r{{{{3, 6}, {4, 8}}}};

    validate_multiplication(a, b, r);
}

/* A: 2 X 3  B:3 X 3 R: 3 * 3 */
TEST(Multiplication, 3x3_3X3_matrices)
{
    Matrix<3, 3> a{{{{1, 2, 3}, {2, 1, 3}, {3, 2, 1}}}};
    Matrix<3, 3> b{{{{1, 1, 1}, {3, 1, 1}, {4, 2, 5}}}};
    Matrix<3, 3> r{{{{19, 9, 18}, {17, 9, 18}, {13, 7, 10}}}};

    validate_multiplication(a, b, r);
}

/* precision is 0.0000001 */
template <std::size_t R, std::size_t C>
void validate_double_matrix(const Matrix<R, C> &r, const Matrix<R, C> &r_expected)
{
    for (std::size_t i = 0; i < R; i++)
    {
        for (std::size_t j = 0; j < C; j++)
        {
            EXPECT_NEAR(r.data()[i][j], r_expected.data()[i][j], 0.0000001);
        }
    }
}

template <std::size_t R, std::size_t C, std::size_t C2>
void validate_m_n_matrix()
{
    Matrix<R, C> a{};
    fill_matrix<double>(a);
    Matrix<C, C2> b{};
    fill_matrix<double>(b);

    auto r = a * b;
    validate_double_matrix<R, C2>(a.multiplication_naive(b), r);
    validate_double_matrix<R, C2>(a.multiplication_t1(b), r);
    validate_double_matrix<R, C2>(a.multiplication_tn(b), r);
    validate_double_matrix<R, C2>(a.multiplication_omp(b), r);
}

TEST(Multiplication, mxn_nXm_matrices)
{
    /* NxN */
    validate_m_n_matrix<3, 3, 3>();
    validate_m_n_matrix<10, 10, 10>();
    validate_m_n_matrix<100, 100, 100>();
    validate_m_n_matrix<250, 250, 250>();
    /* M x N*/
    validate_m_n_matrix<10, 5, 10>();
    validate_m_n_matrix<9, 5, 7>();
    validate_m_n_matrix<120, 5, 150>();
    validate_m_n_matrix<250, 100, 102>();
}

/* Solution: A . B + C = R */
/* To verify the correctness of the optimized expression R = A.B + C R is compared aginst the values from Matrix Class */
template <typename A, typename B, typename C>
void validate_ab_c(A &a, B &b, C &c)
{
    auto r_expected = (a * b) + c; /* Matrix operations are already validated. */

    EXPECT_EQ(ab_c_generic(a, b, c), r_expected);
    EXPECT_EQ(ab_c_optimised(a, b, c), r_expected);
    EXPECT_EQ(ab_c_optimised_tn(a, b, c), r_expected);
    EXPECT_EQ(ab_c_omp(a, b, c), r_expected);
}

TEST(Solution, 3x3_3X3_matrices)
{
    Matrix<3, 3> a{{{{1, 2, 3}, {2, 1, 3}, {3, 2, 1}}}};
    Matrix<3, 3> b{{{{1, 1, 3}, {2, 1, 1}, {1, 2, 1}}}};
    Matrix<3, 3> c{{{{3, 1, 3}, {2, 3, 1}, {1, 2, 3}}}};

    validate_ab_c(a, b, c);
}

/* To verify the correctness of the optimized expression R = A.B + C R is compared aginst the values from Matrix Class */
template <std::size_t R, std::size_t C, std::size_t C2>
void validate_m_n_ab_c()
{
    Matrix<R, C> a{};
    fill_matrix<double>(a);
    Matrix<C, C2> b{};
    fill_matrix<double>(b);

    Matrix<R, C2> c{};
    fill_matrix<double>(c);

    auto r = (a * b) + c;
    validate_double_matrix<R, C2>(ab_c_generic(a, b, c), r);
    validate_double_matrix<R, C2>(ab_c_optimised(a, b, c), r);
    validate_double_matrix<R, C2>(ab_c_optimised_tn(a, b, c), r);
    validate_double_matrix<R, C2>(ab_c_omp(a, b, c), r);
}

TEST(Solution, mxn_nXm_matrices)
{
    /* NxN */
    validate_m_n_ab_c<3, 3, 3>();
    validate_m_n_ab_c<10, 10, 10>();
    validate_m_n_ab_c<100, 100, 100>();
    validate_m_n_ab_c<250, 250, 250>();
    /* M x N*/
    validate_m_n_ab_c<3, 2, 4>();
    validate_m_n_ab_c<9, 5, 7>();
    validate_m_n_ab_c<120, 5, 150>();
    validate_m_n_ab_c<250, 100, 102>();
}

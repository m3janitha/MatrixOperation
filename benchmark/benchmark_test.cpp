#include <benchmark/benchmark.h>
#include <matrix_copy.h>
#include <random>
#include <ctime>
#include <iostream>

template <typename T, typename M>
void fill_matrix(M &m)
{
    /* Create our random number generators */
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<T> dist(-10, 10);

    for (std::size_t r = 0; r < m.rows(); r++)
    {
        for (std::size_t c = 0; c < m.columns(); c++)
        {
            m.data_[r][c] = dist(rng);
        }
    }
}

template <typename MatrixType>
class MatrixFixture : public benchmark::Fixture
{
public:
    void SetUp(::benchmark::State &state) override
    {
        fill_matrix<double>(m1);
        fill_matrix<double>(m2);
        fill_matrix<double>(m3);
    }

    void TearDown(::benchmark::State &state) override
    {
    }

    /* size (M,N) is configured from the test */
    MatrixType m1{};
    MatrixType m2{};
    MatrixType m3{};
};

/* To avoid code duplication */
#define BenchmarkTemplateMatrix(ClassName, FunctionName) \
    BENCHMARK_F(ClassName, BM_##FunctionName)            \
    (benchmark::State & state)                           \
    {                                                    \
        FunctionName(*this, state);                      \
    }

#define BenchmarkTemplateMatrixForAll(ClassName, FunctionName) \
    BenchmarkTemplateMatrix(ClassName##8, FunctionName);       \
    BenchmarkTemplateMatrix(ClassName##16, FunctionName);      \
    BenchmarkTemplateMatrix(ClassName##32, FunctionName);      \
    BenchmarkTemplateMatrix(ClassName##64, FunctionName);      \
    BenchmarkTemplateMatrix(ClassName##128, FunctionName);     \
    BenchmarkTemplateMatrix(ClassName##256, FunctionName);     \
    BenchmarkTemplateMatrix(ClassName##512, FunctionName);
/*BenchmarkTemplateMatrix(##ClassName1024, FunctionName);*/

/* benchmark matrix multiplication */

template <typename Fixture>
static void matrix_multiplication_naive(Fixture &fixture, benchmark::State &state)
{
    for (auto _ : state)
    {
        auto m = fixture.m1 * fixture.m2;
        benchmark::DoNotOptimize(m);
    }
}

using namespace matrix;

using MatrixFixture8 = MatrixFixture<Matrix<8, 8>>;
using MatrixFixture16 = MatrixFixture<Matrix<16, 16>>;
using MatrixFixture32 = MatrixFixture<Matrix<32, 32>>;
using MatrixFixture64 = MatrixFixture<Matrix<64, 64>>;
using MatrixFixture128 = MatrixFixture<Matrix<128, 128>>;
using MatrixFixture256 = MatrixFixture<Matrix<256, 256>>;
using MatrixFixture512 = MatrixFixture<Matrix<512, 512>>;
// using Fixture1024 =  MatrixFixture<Matrix<1024,1024>>;

BenchmarkTemplateMatrixForAll(MatrixFixture, matrix_multiplication_naive);

template <typename Fixture>
static void matrix_multi_optimized_t_1(Fixture & fixture, benchmark::State & state)
{
    for (auto _ : state)
    {
        auto m = fixture.m1 * fixture.m2;
        benchmark::DoNotOptimize(m);
    }
}

using MatrixFixtureT8 = MatrixFixture<MatrixT1<8, 8>>;
using MatrixFixtureT16 = MatrixFixture<MatrixT1<16, 16>>;
using MatrixFixtureT32 = MatrixFixture<MatrixT1<32, 32>>;
using MatrixFixtureT64 = MatrixFixture<MatrixT1<64, 64>>;
using MatrixFixtureT128 = MatrixFixture<MatrixT1<128, 128>>;
using MatrixFixtureT256 = MatrixFixture<MatrixT1<256, 256>>;
using MatrixFixtureT512 = MatrixFixture<MatrixT1<512, 512>>;

BenchmarkTemplateMatrixForAll(MatrixFixtureT, matrix_multi_optimized_t_1);


template <typename Fixture>
static void matrix_multi_optimized_t_n(Fixture &fixture, benchmark::State &state)
{
    for (auto _ : state)
    {
        auto m = fixture.m1 * fixture.m2;
        benchmark::DoNotOptimize(m);
    }
}

using MatrixFixtureN8 = MatrixFixture<MatrixTN<8, 8>>;
using MatrixFixtureN16 = MatrixFixture<MatrixTN<16, 16>>;
using MatrixFixtureN32 = MatrixFixture<MatrixTN<32, 32>>;
using MatrixFixtureN64 = MatrixFixture<MatrixTN<64, 64>>;
using MatrixFixtureN128 = MatrixFixture<MatrixTN<128, 128>>;
using MatrixFixtureN256 = MatrixFixture<MatrixTN<256, 256>>;
using MatrixFixtureN512 = MatrixFixture<MatrixTN<512, 512>>;

BenchmarkTemplateMatrixForAll(MatrixFixtureN, matrix_multi_optimized_t_n)

template <typename Fixture>
static void matrix_abc(Fixture &fixture, benchmark::State &state)
{
    for (auto _ : state)
    {
        auto m = fixture.m1 * fixture.m2 + fixture.m3;
        benchmark::DoNotOptimize(m);
    }
}

using MatrixFixtureABC8 = MatrixFixture<MatrixT1<8, 8>>;
using MatrixFixtureABC16 = MatrixFixture<MatrixT1<16, 16>>;
using MatrixFixtureABC32 = MatrixFixture<MatrixT1<32, 32>>;
using MatrixFixtureABC64 = MatrixFixture<MatrixT1<64, 64>>;
using MatrixFixtureABC128 = MatrixFixture<MatrixT1<128, 128>>;
using MatrixFixtureABC256 = MatrixFixture<MatrixT1<256, 256>>;
using MatrixFixtureABC512 = MatrixFixture<MatrixT1<512, 512>>;

BenchmarkTemplateMatrixForAll(MatrixFixtureABC, matrix_abc)

template <typename Fixture>
static void matrix_abc_f(Fixture &fixture, benchmark::State &state)
{
    for (auto _ : state)
    {
        auto m = ab_c(fixture.m1, fixture.m2, fixture.m3);
        benchmark::DoNotOptimize(m);
    }
}

using MatrixFixtureABCF8 = MatrixFixture<MatrixT1<8, 8>>;
using MatrixFixtureABCF16 = MatrixFixture<MatrixT1<16, 16>>;
using MatrixFixtureABCF32 = MatrixFixture<MatrixT1<32, 32>>;
using MatrixFixtureABCF64 = MatrixFixture<MatrixT1<64, 64>>;
using MatrixFixtureABCF128 = MatrixFixture<MatrixT1<128, 128>>;
using MatrixFixtureABCF256 = MatrixFixture<MatrixT1<256, 256>>;
using MatrixFixtureABCF512 = MatrixFixture<MatrixT1<512, 512>>;

BenchmarkTemplateMatrixForAll(MatrixFixtureABCF, matrix_abc_f)

// /* benchmark matrix addition */

// template <typename Fixture>
// static void matrix_addition(Fixture &fixture, benchmark::State &state)
// {
//     for (auto _ : state)
//     {
//         auto m = fixture.m1 + fixture.m2;
//         benchmark::DoNotOptimize(m);
//     }
// }

// BenchmarkTemplateMatrix(MatrixFixture8, matrix_addition);
// BenchmarkTemplateMatrix(MatrixFixture16, matrix_addition);
// BenchmarkTemplateMatrix(MatrixFixture32, matrix_addition);
// BenchmarkTemplateMatrix(MatrixFixture64, matrix_addition);

// /* benchmark matrix subtraction */

// template <typename Fixture>
// static void matrix_subtraction(Fixture &fixture, benchmark::State &state)
// {
//     for (auto _ : state)
//     {
//         auto m = fixture.m1 - fixture.m2;
//         benchmark::DoNotOptimize(m);
//     }
// }

// BenchmarkTemplateMatrix(MatrixFixture8, matrix_subtraction);
// BenchmarkTemplateMatrix(MatrixFixture16, matrix_subtraction);
// BenchmarkTemplateMatrix(MatrixFixture32, matrix_subtraction);
// BenchmarkTemplateMatrix(MatrixFixture64, matrix_subtraction);

// /* benchmark matrix operations */

// template <typename Fixture>
// static void matrix_operations(Fixture &fixture, benchmark::State &state)
// {
//     for (auto _ : state)
//     {
//         auto m = fixture.m1 * fixture.m2 + fixture.m3;
//         benchmark::DoNotOptimize(m);
//     }
// }

// BenchmarkTemplateMatrix(MatrixFixture8, matrix_operations);
// BenchmarkTemplateMatrix(MatrixFixture16, matrix_operations);
// BenchmarkTemplateMatrix(MatrixFixture32, matrix_operations);
// BenchmarkTemplateMatrix(MatrixFixture64, matrix_operations);

BENCHMARK_MAIN();
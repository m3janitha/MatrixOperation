#include <benchmark/benchmark.h>
#include <matrix_operations/matrix.h>
#include <matrix_operations/solution.h>
#include <matrix_operations/matrix_util.h>
#include <iostream>

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

using namespace matrix;

using MatrixFixture8 = MatrixFixture<Matrix<8, 8>>;
using MatrixFixture16 = MatrixFixture<Matrix<16, 16>>;
using MatrixFixture32 = MatrixFixture<Matrix<32, 32>>;
using MatrixFixture64 = MatrixFixture<Matrix<64, 64>>;
using MatrixFixture128 = MatrixFixture<Matrix<128, 128>>;
using MatrixFixture256 = MatrixFixture<Matrix<256, 256>>;
using MatrixFixture512 = MatrixFixture<Matrix<512, 512>>;
using MatrixFixture1024 = MatrixFixture<Matrix<1024, 1024>>;
using MatrixFixture2048 = MatrixFixture<Matrix<2048, 2048>>;

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

#define BenchmarkTemplateMatrixForAll_BIG(ClassName, FunctionName) \
    BenchmarkTemplateMatrixForAll(ClassName, FunctionName);        \
    BenchmarkTemplateMatrix(ClassName##1024, FunctionName);        \
    BenchmarkTemplateMatrix(ClassName##2048, FunctionName);

/* benchmark matrix multiplication */

template <typename Fixture>
static void matrix_multiplication_naive(Fixture &fixture, benchmark::State &state)
{
    for (auto _ : state)
    {
        auto m = fixture.m1.multiplication_naive(fixture.m2);
        benchmark::DoNotOptimize(m);
    }
}

BenchmarkTemplateMatrixForAll(MatrixFixture, matrix_multiplication_naive);

template <typename Fixture>
static void matrix_multiplication_t1(Fixture &fixture, benchmark::State &state)
{
    for (auto _ : state)
    {
        auto m = fixture.m1.multiplication_t1(fixture.m2);
        benchmark::DoNotOptimize(m);
    }
}

BenchmarkTemplateMatrixForAll(MatrixFixture, matrix_multiplication_t1);

template <typename Fixture>
static void matrix_multiplication_blocked(Fixture &fixture, benchmark::State &state)
{
    for (auto _ : state)
    {
        auto m = fixture.m1.multiplication_blocked(fixture.m2);
        benchmark::DoNotOptimize(m);
    }
}

BenchmarkTemplateMatrixForAll(MatrixFixture, matrix_multiplication_blocked);

template <typename Fixture>
static void matrix_multiplication_tn(Fixture &fixture, benchmark::State &state)
{
    for (auto _ : state)
    {
        auto m = fixture.m1.multiplication_tn(fixture.m2);
        benchmark::DoNotOptimize(m);
    }
}

BenchmarkTemplateMatrixForAll(MatrixFixture, matrix_multiplication_tn);

template <typename Fixture>
static void matrix_multiplication_omp(Fixture &fixture, benchmark::State &state)
{
    for (auto _ : state)
    {
        auto m = fixture.m1.multiplication_omp(fixture.m2);
        benchmark::DoNotOptimize(m);
    }
}

BenchmarkTemplateMatrixForAll(MatrixFixture, matrix_multiplication_omp);

template <typename Fixture>
static void matrix_multiplication_operator(Fixture &fixture, benchmark::State &state)
{
    for (auto _ : state)
    {
        auto m = fixture.m1 * fixture.m2;
        benchmark::DoNotOptimize(m);
    }
}

BenchmarkTemplateMatrixForAll(MatrixFixture, matrix_multiplication_operator);

/* Addition */
template <typename Fixture>
static void matrix_addition(Fixture &fixture, benchmark::State &state)
{
    for (auto _ : state)
    {
        auto m = fixture.m1.addition(fixture.m2);
        benchmark::DoNotOptimize(m);
    }
}

BenchmarkTemplateMatrixForAll(MatrixFixture, matrix_addition);

template <typename Fixture>
static void matrix_addition_tn(Fixture &fixture, benchmark::State &state)
{
    for (auto _ : state)
    {
        auto m = fixture.m1.addition_tn(fixture.m2);
        benchmark::DoNotOptimize(m);
    }
}

/* Solution */
BenchmarkTemplateMatrixForAll(MatrixFixture, matrix_addition_tn);

template <typename Fixture>
static void BM_ab_c_generic(Fixture &fixture, benchmark::State &state)
{
    for (auto _ : state)
    {
        auto m = ab_c_generic(fixture.m1, fixture.m2, fixture.m3);
        benchmark::DoNotOptimize(m);
    }
}

BenchmarkTemplateMatrixForAll(MatrixFixture, BM_ab_c_generic);

template <typename Fixture>
static void BM_ab_c_optimised(Fixture &fixture, benchmark::State &state)
{
    for (auto _ : state)
    {
        auto m = ab_c_optimised(fixture.m1, fixture.m2, fixture.m3);
        benchmark::DoNotOptimize(m);
    }
}

BenchmarkTemplateMatrixForAll(MatrixFixture, BM_ab_c_optimised);

template <typename Fixture>
static void BM_ab_c_optimised_tn(Fixture &fixture, benchmark::State &state)
{
    for (auto _ : state)
    {
        auto m = ab_c_optimised_tn(fixture.m1, fixture.m2, fixture.m3);
        benchmark::DoNotOptimize(m);
    }
}

BenchmarkTemplateMatrixForAll(MatrixFixture, BM_ab_c_optimised_tn);

template <typename Fixture>
static void BM_ab_c_omp(Fixture &fixture, benchmark::State &state)
{
    for (auto _ : state)
    {
        auto m = ab_c_omp(fixture.m1, fixture.m2, fixture.m3);
        benchmark::DoNotOptimize(m);
    }
}

BenchmarkTemplateMatrixForAll(MatrixFixture, BM_ab_c_omp);

template <typename Fixture>
static void BM_ab_c(Fixture &fixture, benchmark::State &state)
{
    for (auto _ : state)
    {
        auto m = ab_c(fixture.m1, fixture.m2, fixture.m3);
        benchmark::DoNotOptimize(m);
    }
}

BenchmarkTemplateMatrixForAll(MatrixFixture, BM_ab_c);

BENCHMARK_MAIN();
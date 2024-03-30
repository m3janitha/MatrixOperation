#include <benchmark/benchmark.h>
#include <matrix.h>
#include <cstdlib>
#include <ctime>
#include <iostream>

template <typename M>
void fill_matrix(M &m)
{
    std::srand(std::time(nullptr));

    for (std::size_t r = 0; r < m.rows(); r++)
    {
        for (std::size_t c = 0; c < m.columns(); c++)
        {
            m.data_[r][c] = std::rand();
        }
    }
}

template <std::size_t N>
class MatrixFixture : public benchmark::Fixture
{
public:
    void SetUp(::benchmark::State &state) override
    {
        fill_matrix(m1);
    }

    void TearDown(::benchmark::State &state) override
    {
    }

    Matrix<N, N, int> m1{};
    Matrix<N, N, int> m2{};
    Matrix<N, N, int> m3{};
};

using MatrixFixture8 = MatrixFixture<8>;
using MatrixFixture16 = MatrixFixture<16>;
using MatrixFixture32 = MatrixFixture<32>;
using MatrixFixture64 = MatrixFixture<64>;

/* to avoid code duplication */
#define BenchmarkTemplateMatrix(ClassName, FunctionName) \
BENCHMARK_F(ClassName, BM_##FunctionName)(benchmark::State &state) \
{ FunctionName(*this, state); }   

/* benchmark matrix multiplication */

template <typename Fixture>
static void matrix_multiplication(Fixture &fixture, benchmark::State &state)
{
    for (auto _ : state)
    {
        auto m = fixture.m1 * fixture.m2;
        benchmark::DoNotOptimize(m);
    }
}                                                 

BenchmarkTemplateMatrix(MatrixFixture8, matrix_multiplication);
BenchmarkTemplateMatrix(MatrixFixture16, matrix_multiplication);
BenchmarkTemplateMatrix(MatrixFixture32, matrix_multiplication);
BenchmarkTemplateMatrix(MatrixFixture64, matrix_multiplication);

/* benchmark matrix addition */

template <typename Fixture>
static void matrix_addition(Fixture &fixture, benchmark::State &state)
{
    for (auto _ : state)
    {
        auto m = fixture.m1 + fixture.m2;
        benchmark::DoNotOptimize(m);
    }
}

BenchmarkTemplateMatrix(MatrixFixture8, matrix_addition);
BenchmarkTemplateMatrix(MatrixFixture16, matrix_addition);
BenchmarkTemplateMatrix(MatrixFixture32, matrix_addition);
BenchmarkTemplateMatrix(MatrixFixture64, matrix_addition);

/* benchmark matrix subtraction */

template <typename Fixture>
static void matrix_subtraction(Fixture &fixture, benchmark::State &state)
{
    for (auto _ : state)
    {
        auto m = fixture.m1 - fixture.m2;
        benchmark::DoNotOptimize(m);
    }
}

BenchmarkTemplateMatrix(MatrixFixture8, matrix_subtraction);
BenchmarkTemplateMatrix(MatrixFixture16, matrix_subtraction);
BenchmarkTemplateMatrix(MatrixFixture32, matrix_subtraction);
BenchmarkTemplateMatrix(MatrixFixture64, matrix_subtraction);

/* benchmark matrix operations */

template <typename Fixture>
static void matrix_operations(Fixture &fixture, benchmark::State &state)
{
    for (auto _ : state)
    {
        auto m = fixture.m1 * fixture.m2 + fixture.m3;
        benchmark::DoNotOptimize(m);
    }
}

BenchmarkTemplateMatrix(MatrixFixture8, matrix_operations);
BenchmarkTemplateMatrix(MatrixFixture16, matrix_operations);
BenchmarkTemplateMatrix(MatrixFixture32, matrix_operations);
BenchmarkTemplateMatrix(MatrixFixture64, matrix_operations);

BENCHMARK_MAIN();
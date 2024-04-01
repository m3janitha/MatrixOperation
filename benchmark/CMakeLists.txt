include_directories(${PROJECT_SOURCE_DIR})
message("PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}")

set(BENCHMARK_ENABLE_TESTING OFF)

FetchContent_Declare(
	googlebenchmark
	GIT_REPOSITORY https://github.com/google/benchmark
	GIT_TAG v1.6.0
)

FetchContent_MakeAvailable(googlebenchmark)

add_executable(benchmark_matrix benchmark_matrix.cpp)
target_link_libraries(benchmark_matrix benchmark::benchmark)
target_link_libraries(benchmark_matrix GTest::gtest GTest::gtest_main)

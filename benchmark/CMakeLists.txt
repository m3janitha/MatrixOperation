include_directories(${PROJECT_SOURCE_DIR}/matrix_operations)

set(BENCHMARK_ENABLE_TESTING OFF)

FetchContent_Declare(
	googlebenchmark
	GIT_REPOSITORY https://github.com/google/benchmark
	GIT_TAG v1.6.0
)

FetchContent_MakeAvailable(googlebenchmark)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message("setting MSVC CMAKE_CXX_FLAGS")
    target_compile_options(matrix_operations PRIVATE /W4 /WX)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message("setting GCC CMAKE_CXX_FLAGS")
    target_compile_options(matrix_operations PRIVATE -Wall -Wextra -pedantic -Werror -Wno-unused-but-set-variable -msse2 -mfpmath=sse -ftree-vectorizer-verbose=5)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(matrix_operations PRIVATE -Wno-unused-variable)    
else()
    message("setting ${CMAKE_CXX_COMPILER_ID} CMAKE_CXX_FLAGS")
endif()

add_executable(benchmark_test benchmark_matrix.cpp)
target_link_libraries(benchmark_test benchmark::benchmark)
target_link_libraries(benchmark_test GTest::gtest GTest::gtest_main)



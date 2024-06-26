cmake_minimum_required (VERSION 3.16)

project(MatrixOperations VERSION 1.0)

set(CMAKE_CPP_COMPILER g++)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_VERBOSE_MAKEFILE ON)

# lots of warnings and all warnings as errors
message("setting CMAKE_CXX_FLAGS")
set(CMAKE_BUILD_TYPE Release)
    
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message("setting MSVC CMAKE_CXX_FLAGS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ox")
    add_compile_options(/wd2220)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message("setting GCC CMAKE_CXX_FLAGS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mfpmath=sse -ftree-vectorizer-verbose=5 -ffast-math -fopenmp")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message("setting Clang CMAKE_CXX_FLAGS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")    
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
else()
    message("setting ${CMAKE_CXX_COMPILER_ID} CMAKE_CXX_FLAGS")
endif()

include(FetchContent)

FetchContent_Declare(
	googletest
	GIT_REPOSITORY https://github.com/google/googletest.git
	GIT_TAG release-1.12.1
)

FetchContent_MakeAvailable(googletest)

include(GoogleTest)
enable_testing()

add_subdirectory(matrix_operations)
add_subdirectory(benchmark)
add_subdirectory(test)

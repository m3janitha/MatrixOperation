include_directories(${PROJECT_SOURCE_DIR})

add_executable (matrix_operations main.cpp)

target_include_directories(matrix_operations PRIVATE "/usr/include")

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message("setting MSVC CMAKE_CXX_FLAGS")
    target_compile_options(matrix_operations PRIVATE /W4 /WX)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message("setting GCC CMAKE_CXX_FLAGS")
    target_compile_options(matrix_operations PRIVATE -Wall -Wextra -pedantic -Werror)
    target_compile_options(matrix_operations PRIVATE -Wno-unused-variable) 
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(matrix_operations PRIVATE -Wno-unused-variable)    
else()
    message("setting ${CMAKE_CXX_COMPILER_ID} CMAKE_CXX_FLAGS")
endif()
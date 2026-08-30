cmake_minimum_required(VERSION 3.15)
project(BalanceNN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Include directories ---
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/thirdparty
)

# --- Main executable ---
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/BalanceNN.cpp
    src/nn.cpp
)

# --- Tests ---
enable_testing()

add_executable(test_matrix src/test_matrix.cpp)
add_test(NAME matrix COMMAND test_matrix)

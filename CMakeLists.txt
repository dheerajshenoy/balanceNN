cmake_minimum_required(VERSION 3.24)
project(BalanceNN LANGUAGES CXX)

# Use timestamps from extraction (not the archive) so URL changes trigger rebuilds.
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- LibTorch (auto-download prebuilt) ---
# Override with -DLIBTORCH_VARIANT=cu121 (or cu118 etc.) for CUDA builds,
# or -DLIBTORCH_VERSION=2.5.1 to pin a specific version.
set(LIBTORCH_VERSION "2.5.1" CACHE STRING "libtorch version to download")
set(LIBTORCH_VARIANT "cpu"   CACHE STRING "libtorch build variant (cpu, cu118, cu121, ...)")

if(WIN32)
    set(LIBTORCH_URL
        "https://download.pytorch.org/libtorch/${LIBTORCH_VARIANT}/libtorch-win-shared-with-deps-${LIBTORCH_VERSION}%2B${LIBTORCH_VARIANT}.zip")
elseif(APPLE)
    # macOS wheels are CPU-only and use a different filename.
    set(LIBTORCH_URL
        "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${LIBTORCH_VERSION}.zip")
else()
    set(LIBTORCH_URL
        "https://download.pytorch.org/libtorch/${LIBTORCH_VARIANT}/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${LIBTORCH_VARIANT}.zip")
endif()

include(FetchContent)
FetchContent_Declare(
    libtorch
    URL       ${LIBTORCH_URL}
    SOURCE_DIR ${CMAKE_BINARY_DIR}/libtorch
)
FetchContent_MakeAvailable(libtorch)

list(APPEND CMAKE_PREFIX_PATH "${libtorch_SOURCE_DIR}")
find_package(Torch REQUIRED)

# --- SDL3 (fetched from source) ---
set(SDL3_VERSION "3.2.24" CACHE STRING "SDL3 release tag to fetch")

set(SDL_SHARED   ON  CACHE BOOL "" FORCE)
set(SDL_STATIC   OFF CACHE BOOL "" FORCE)
set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    SDL3
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG        release-${SDL3_VERSION}
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(SDL3)

# --- Include directories ---
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/thirdparty
)

# --- Main executable ---
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/BalanceNN.cpp
)
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} SDL3::SDL3)
target_compile_options(${PROJECT_NAME} PRIVATE ${TORCH_CXX_FLAGS})

# --- Tests ---
# enable_testing()
# add_executable(test_matrix src/test_matrix.cpp)
# target_link_libraries(test_matrix PRIVATE ${TORCH_LIBRARIES})
# target_compile_options(test_matrix PRIVATE ${TORCH_CXX_FLAGS})
# add_test(NAME matrix COMMAND test_matrix)

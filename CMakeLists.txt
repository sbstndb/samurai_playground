cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-mtune=native -march=native -O3 -mavx2 -mfma")

project(playground CXX)

find_package(xtensor REQUIRED)
find_package(samurai CONFIG REQUIRED)


add_subdirectory(src/vector_add)

set(KERNEL "/home/sbstndbs/samurai_playground/build/src/vector_add/vector_add")
add_subdirectory(profilers/perf)
add_subdirectory(profilers/maqao)


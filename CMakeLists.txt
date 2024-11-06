cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(playground CXX)

find_package(xtensor REQUIRED)
find_package(samurai CONFIG REQUIRED)

add_subdirectory(src/vector_add)


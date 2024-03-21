cmake_minimum_required(VERSION 3.26)
project(optimised)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP) # OpenMP 찾기
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(optimised mega-stream.cpp)

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(ukf)

# Executable for Unscented Kalman Filter exercise
include_directories(include ${CMAKE_CURRENT_SOURCE_DIR}/../Eigen)
add_executable (ukf main.cpp src/ukf.cpp)
target_link_libraries (ukf)

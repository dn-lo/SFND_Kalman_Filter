cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(lidar)

# Executable for lidar tracking exercise
include_directories(include ${CMAKE_CURRENT_SOURCE_DIR}/../Eigen)
add_executable (lidar_tracking main.cpp src/kalman_filter.cpp src/tracking.cpp)
target_link_libraries (lidar_tracking)
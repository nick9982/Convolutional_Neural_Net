cmake_minimum_required(VERSION 3.16)

add_library(tests updateTestDenseLayer.cpp)

target_include_directories(${This} PRIVATE ./)

cmake_minimum_required(VERSION 3.16)

add_library(process_power_consumption power_consumption.cpp)

target_include_directories(${This} PRIVATE ./)

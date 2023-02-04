cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)
set(This cnn)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -g")

project(${This} CXX C)

include_directories(
    include
    /usr/include
)

add_executable(${PROJECT_NAME} mnist_images.hpp nnalgorithms.hpp NeuralNetwork.hpp NeuralNetwork.cpp mnist_images.cpp DenseLayer.cpp ConvolutionalLayer.cpp PoolingLayer.cpp randomize_seed.cpp nnalgorithms.cpp initializer.cpp main.cpp)

add_subdirectory(DataMining)

target_link_libraries(${PROJECT_NAME} process_power_consumption)

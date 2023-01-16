cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)
set(This cnn)

project(${This} CXX C)

include_directories(
    include
    /usr/include
)

add_executable(${PROJECT_NAME} mnist_images.hpp nnalgorithms.hpp NeuralNetwork.hpp NeuralNetwork.cpp mnist_images.cpp DenseLayer.cpp ConvolutionalLayer.cpp PoolingLayer.cpp Neuron.cpp Weight.cpp randomize_seed.cpp nnalgorithms.cpp StorageForNeuronsAndWeights.cpp Bias.cpp main.cpp)

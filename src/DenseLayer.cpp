#include "NeuralNetwork.hpp"

DenseLayer::DenseLayer(int input_size, string activation, string initialization)
{
    this->in = input_size;
    this->activation = stringActivationToIntActivation(activation);
    this->initialization = stringInitializationToIntInitialization(initialization);
}

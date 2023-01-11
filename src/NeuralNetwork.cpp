#include "NeuralNetwork.hpp"

int stringActivationToIntActivation(string activation_function)
{
    if(activation_function == "Linear" || activation_function == "") return 0;
    if(activation_function == "ReLU") return 1;
    return 0;
}

int stringInitializationToIntInitialization(string initialization)
{
    if(initialization == "Zero" || initialization == "") return 0;
    if(initialization == "Xavier") return 1;
    return 0;
}

void NeuralNetwork::initialize()
{
    cout << "runs" << endl;
}

#include "NeuralNetwork.hpp"
#include "nnalgorithms.hpp"

Neuron::Neuron(int activation)
{
    //setting the function pointers to a variable for the activation function
    switch(activation) 
    {
        case 0:
            this->act_function = &ReLU;
            this->act_function_derivative = &ReLUDerivative;
            break;
    }
}

void Neuron::set_value(double value)
{
    this->value = value;
}

double Neuron::get_value()
{
    return this->value;
}


void Neuron::set_value_and_activate(double val)
{
    this->cache_value = val;
    this->act_function(val);
    cout << val << endl;
}

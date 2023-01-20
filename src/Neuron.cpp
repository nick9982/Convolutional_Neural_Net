#include "NeuralNetwork.hpp"
#include "nnalgorithms.hpp"

Neuron::Neuron(int activation)
{
    //setting the function pointers to a variable for the activation function
    switch(activation) 
    {
        case 0:
            this->act_function = Linear;
            this->act_function_derivative = LinearDerivative;
            break;
        case 1:
            this->act_function = ReLU;
            this->act_function_derivative = ReLUDerivative;
            break;
    }
}

void Neuron::set_value(double value)
{
    this->value = value;
}

void Neuron::set_cache(double cache)
{
    this->cache_value = cache;
}

double Neuron::get_value()
{
    return this->value;
}


void Neuron::set_value_and_activate(double val)
{
    this->cache_value = val;
    this->value = this->act_function(val);
}

double Neuron::get_derivative()
{
    return this->act_function_derivative(this->cache_value);
}

void Neuron::set_delta(double value)
{
    this->delta_value = value;
}

double Neuron::get_delta()
{
    return this->delta_value;
}

#include "NeuralNetwork.hpp"

Bias::Bias(bool exists)
{
    this->value = 0;
    this->exists = exists;
    this->m = 0;
    this->v = 0;
    this->alpha = 0.001;
}

void Bias::set_value(double value)
{
    this->value = value;
}

double Bias::get_value()
{
    return this->value;
}

void Bias::set_exists(bool exists)
{
    this->exists = exists;
}

bool Bias::get_exists()
{
    return this->exists;
}

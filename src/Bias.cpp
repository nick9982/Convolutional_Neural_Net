#include "NeuralNetwork.hpp"

Bias::Bias(bool exists)
{
    this->value = 0;
    this->exists = exists;
    this->m = 0;
    this->v = 0;
    this->alpha = learningRate;
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

void Bias::update(double gradient)
{
    switch(optimizer)
    {
        case 0:
            this->value -= learningRate * gradient;
            break;
        case 1:
            this->m = beta1 * this->m + (1 - beta1) * gradient;
            this->v = beta2 * this->v + (1 - beta2) * pow(gradient, 2);

            double mhat = this->m / (1 - pow(beta1, epoch));
            double vhat = this->v / (1 - pow(beta2, epoch));

            this->value -= this->alpha * gradient;
    }
}

#include "NeuralNetwork.hpp"
#include "nnalgorithms.hpp"

double beta1 = 0.9;
double beta2 = 0.999;

Weight::Weight(int in, int out, int initialization)
{
    this->input = in;
    this->output = out;
    this->initialization = initialization;
    this->alpha = learningRate;
    this->init();
}

void Weight::init()
{
    switch(this->initialization)
    {
        case 0:
            this->value = HeRandomInNormal(this->input);
            break;
        case 1:
            this->value = HeRandomAvgNormal(this->input, this->output);
            break;
        case 2:
            this->value = HeRandomInUniform(this->input);
            break;
        case 3:
            this->value = HeRandomAvgUniform(this->input, this->output);
            break;
        case 4:
            this->value = XavierRandomNormal(this->input, this->output);
            break;
        case 5:
            this->value = XavierRandomUniform(this->input, this->output);
            break;
        case 6:
            this->value = LeCunRandom(this->input);
            break;
        case 12:
            this->value = 0.5;
            break;
        default:
            this->value = 0;
            break;
    }
}

double Weight::get_value()
{
    return this->value;
}

void Weight::set_value(double value)
{
    this->value = value;
}

void Weight::update(double gradient)
{
    switch(optimizer)
    {
        case 0:
            this->value -= learningRate * gradient;
            break;
        case 1:
            this->m = beta1 * this->m + (1 - beta1) * gradient;
            this->v = beta2 * this->v + (1 - beta2) * pow(gradient, 2);
    
            double mhat = this->m / (1-pow(beta1, epoch));
            double vhat = this->v / (1-pow(beta2, epoch));
    
            this->value -= (this->alpha / (sqrt(vhat + 1e-8)) * mhat);
            break;
    }
}

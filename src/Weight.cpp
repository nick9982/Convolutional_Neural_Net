#include "NeuralNetwork.hpp"
#include "nnalgorithms.hpp"

Weight::Weight(int in, int out, int initialization)
{
    this->input = in;
    this->output = out;
    this->initialization = initialization;
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
        default:
            this->value = 0;
            break;
    }
}

double Weight::get_value()
{
    return this->value;
}

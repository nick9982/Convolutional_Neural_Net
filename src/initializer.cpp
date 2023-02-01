#include "NeuralNetwork.hpp"
#include "nnalgorithms.hpp"

Initializer::Initializer(int init_formula, int in, int out)
{
    this->init_formula = init_formula;
    this->in = in;
    this->out = out;
}

double Initializer::init()
{
    switch(this->init_formula)
    {
        case 0:
            return HeRandomInNormal(this->in);
        case 1:
            return HeRandomAvgNormal(this->in, this->out);
        case 2:
            return HeRandomInUniform(this->in);
        case 3:
            return HeRandomAvgUniform(this->in, this->out);
        case 4:
            return XavierRandomNormal(this->in, this->out);
        case 5:
            return XavierRandomUniform(this->in, this->out);
        case 6:
            return LeCunRandom(this->in);
        case 12:
            return 0.5;
        default:
            return 0;
    }
}

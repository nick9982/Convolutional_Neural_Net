#include "nnalgorithms.hpp"
#include "NeuralNetwork.hpp"

double randomDoubleInRange(double hi, double lo)
{
    double f = (double)rand() / RAND_MAX;
    return lo + f * (hi-lo);
}

/*  Activation functions  */

double ReLU(double input)
{
    if(input > 0) return input;
    return 0;
}

double ReLUDerivative(double input)
{
    if(input <= 0) return 0;
    return 1;
}

double Linear(double input)
{
    return input;
}

double LinearDerivative(double input)
{
    return 1;
}

/*  Initialization functions  */
double HeRandomInNormal(int input)
{
    double hi = sqrt(2/input);
    return randomDoubleInRange(hi, -hi);
}

double HeRandomAvgNormal(int input, int output)
{
    double hi = sqrt(2/((double)(input + output)/2));
    return randomDoubleInRange(hi, -hi);
}

double HeRandomInUniform(int input)
{
    double hi = sqrt(6/input);
    return randomDoubleInRange(hi, -hi);
}

double HeRandomAvgUniform(int input, int output)
{
    double hi = sqrt(6/((double)(input + output)/2));
    return randomDoubleInRange(hi, -hi);
}

double XavierRandomNormal(int input, int output)
{
    double hi = sqrt(1/((double)(input+output)/2));
    return randomDoubleInRange(hi, -hi);
}

double XavierRandomUniform(int input, int output)
{
    double hi = sqrt(3/((double)(input+output)/2));
    return randomDoubleInRange(hi, -hi);
}

double LeCunRandom(int input)
{
    double hi = sqrt(1.0/input);
    return randomDoubleInRange(hi, -hi);
}

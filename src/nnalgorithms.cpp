#include "nnalgorithms.hpp"
#include "NeuralNetwork.hpp"

double randomDoubleDistribution(double hi)
{
    return (double)rand() / RAND_MAX * (hi*2) - hi;
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

double* SoftMax(double *input, int size)
{
    double sum = 0;
    for(int i = 0; i < size; i++)
    {
        cout << "inp: " << input[i] << endl;
        input[i] = exp(input[i]);
        sum += input[i];
    }

    for(int i = 0; i < size; i++)
    {
        input[i] = input[i]/sum;
    }
    return input;
}

double* SoftMaxDerivative(double* input, int size)
{
    return input;
}

/*  Initialization functions  */
double HeRandomInNormal(int input)
{
    return randomDoubleDistribution(sqrt(2/(double)input));
}

double HeRandomAvgNormal(int input, int output)
{
    return randomDoubleDistribution(sqrt(2/((double)(input + output)/2)));
}

double HeRandomInUniform(int input)
{
    return randomDoubleDistribution(sqrt(6/(double)input));
}

double HeRandomAvgUniform(int input, int output)
{
    return randomDoubleDistribution(sqrt(6/((double)(input+output)/2)));
}

double XavierRandomNormal(int input, int output)
{
    return randomDoubleDistribution(sqrt(1/((double)(input+output)/2)));
}

double XavierRandomUniform(int input, int output)
{
    return randomDoubleDistribution(sqrt(3/((double)(input+output)/2)));
}

double LeCunRandom(int input)
{
    return randomDoubleDistribution(sqrt(1.0/input));
}

#include "NeuralNetwork.hpp"
#include <exception>
StorageForNeuronsAndWeights globalStore;
DenseLayer::DenseLayer(int input_size, string activation, string initialization)
{
    this->in = input_size;
    this->activation = stringActivationToIntActivation(activation);
    this->initialization = stringInitializationToIntInitialization(initialization);
}

int DenseLayer::getIn()
{
    return this->in;
}

void DenseLayer::init(int output, int layerType, int idx)
{
    this->out = output;
    this->layerType = layerType;
    this->idx = idx;

    globalStore.createDenseLayer(this->in, this->out, this->activation, this->initialization);
}

void DenseLayer::forward()
{
    vector<double> result;
    for(int i = 0; i < this->out; i++)
    {
        double sum = 0;
        for(int j = 0; j < this->in; j++)
        {
            sum += globalStore.NeuronStore[this->idx][i]->get_value() * globalStore.WeightStore[this->idx][(i*this->in + j)]->get_value();
        }
        globalStore.NeuronStore[this->idx+1][i]->set_value_and_activate(sum);
    }
}

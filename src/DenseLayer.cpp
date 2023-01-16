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

    if(this->layerType == 2)
    {
        this->bias.set_exists(true);
    }
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
            sum += globalStore.NeuronStore[this->idx][j]->get_value() * globalStore.WeightStore[this->idx][(i*this->in + j)]->get_value();
        }
        if(this->bias.get_exists())
        {
            sum += this->bias.get_value();
        }
        globalStore.NeuronStore[this->idx+1][i]->set_value_and_activate(sum);
    }
}

void DenseLayer::firstDeltas(vector<double> errors)
{
    vector<double> result(this->in, 0);
    for(int i=0; i < errors.size(); i++)
    {
        Neuron* neurptr = globalStore.NeuronStore[this->idx][i];
        double delta = (neurptr->get_value() - errors[i]) * neurptr->get_derivative(); 
    }
}

void DenseLayer::backward()
{
    for(int i = 0; i < this->in; i++)
    {
        double neuron_derivative = globalStore.NeuronStore[this->idx][i].get_derivative();
        double sum = 0;
        for(int j = 0; j < this->out; j++)
        {
            sum += globalStore.WeightStore[this->idx][(i*this->out+j)]->get_value() * globalStore.NeuronStore[this->idx+1][j]->get_delta() * neuron_derivative;
        }
        
    }
}

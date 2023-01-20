#include "NeuralNetwork.hpp"
#include <exception>
#include <stdexcept>
StorageForNeuronsAndWeights globalStore;
DenseLayer::DenseLayer(int input_size, string activation, string initialization, bool bias)
{
    this->in = input_size;
    this->activation = stringActivationToIntActivation(activation);
    this->initialization = stringInitializationToIntInitialization(initialization);
    this->hasBias = bias;
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

    globalStore.createDenseLayer(this->in, this->out, 1, this->activation, this->initialization);
    if(this->layerType > 1 && this->hasBias)
    {
        globalStore.BiasStore[this->idx][0]->set_exists(true);
    }
}

void DenseLayer::forward()
{
    for(int i = 0; i < this->out; i++)
    {
        double sum = 0;
        for(int j = 0; j < this->in; j++)
        {
            sum += globalStore.NeuronStore[this->idx][j]->get_value() * 
            globalStore.WeightStore[this->idx][j*this->out+i]->get_value();
        }
        globalStore.NeuronStore[this->idx+1][i]->set_value_and_activate(sum);
    }
}

void DenseLayer::firstDeltas(vector<double> errors)
{
    for(int i = 0; i < this->in; i++)
    {
        globalStore.NeuronStore[this->idx][i]->set_delta(
            (globalStore.NeuronStore[this->idx][i]->get_value() - errors[i]) *
            globalStore.NeuronStore[this->idx][i]->get_derivative()
        );
    }
}

void DenseLayer::backward()
{
    for(int i = 0; i < this->in; i++)
    {
        double sum = 0;
        double neuron_derivative = globalStore.NeuronStore[this->idx][i]->get_derivative();
        for(int j = 0; j < this->out; j++)
        {
            sum += globalStore.WeightStore[this->idx][i*this->out+j]->get_value() *
            globalStore.NeuronStore[this->idx+1][j]->get_delta() * neuron_derivative;
        }
        globalStore.NeuronStore[this->idx][i]->set_delta(sum);
    }
}

void DenseLayer::update()
{
    for(int i = 0; i < this->in; i++)
    {
        for(int j = 0; j < this->out; j++)
        {
            globalStore.WeightStore[this->idx][i*this->out+j]->update(globalStore.NeuronStore[this->idx][i]->get_value()
                * globalStore.NeuronStore[this->idx+1][j]->get_delta());
        }
    }

}

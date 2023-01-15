#include "NeuralNetwork.hpp"

void StorageForNeuronsAndWeights::createDenseLayer(int input, int output, int activation, int initialization)
{
    int noOfWeights = input * output;
    vector<Neuron*> neurs;
    vector<Weight*> wts;

    for(int i = 0; i < input; i++)
    {
        neurs.push_back(new Neuron(activation));
    }

    for(int i = 0; i < noOfWeights; i++)
    {
        wts.push_back(new Weight(input, output, initialization));
    }

    this->NeuronStore.push_back(neurs);
    this->WeightStore.push_back(wts);
}

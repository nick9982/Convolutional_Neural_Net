#include "NeuralNetwork.hpp"
#include <stdexcept>

int stringActivationToIntActivation(string activation_function)
{
    if(activation_function == "Linear" || activation_function == "") return 0;
    if(activation_function == "ReLU") return 1;
    return 0;
}

int stringInitializationToIntInitialization(string initialization)
{
    if(initialization == "Zero" || initialization == "") return 0;
    if(initialization == "Xavier") return 1;
    return 0;
}

vector<Neuron*> translateDoubleVecToNeuronVec(vector<double> vec)
{
    vector<Neuron*> result;
    for(int i = 0; i < vec.size(); i++)
    {
        result.push_back(new Neuron(0));
        result[i]->set_value(vec[i]);
    }
    return result;
}

void NeuralNetwork::createResultBuffer()
{
    if(holds_alternative<DenseLayer*>(this->layers[this->layers.size()-1]))
    {
        int outputNodes = get<DenseLayer*>(this->layers[this->layers.size()-1])->getIn();
        this->result_buf = vector<double>(outputNodes, 0);
    }
}
typedef variant<int, vector<int>> Dimensions;

void NeuralNetwork::initialize()
{
    
    for(int i = 0; i < layers.size(); i++)
    {
        Dimensions output_size = 0;
        if(i != layers.size()-1)
        {
            if(holds_alternative<DenseLayer*>(layers[i+1]))
            {
                output_size = get<DenseLayer*>(layers[i+1])->getIn();
            }
            else if(holds_alternative<ConvolutionalLayer*>(layers[i+1]))
            {
                output_size = get<ConvolutionalLayer*>(layers[i+1])->getIn();
            }
            else
            {
                /* output_size = get<PoolingLayer*>(layers[i+1])->getIn(); */
            }
        }
        int layerType = 2; //input - 0, output - 1, hidden - 2
        if(i == 0) layerType = 0;
        if(i == layers.size()-1) layerType = 1;
        if(holds_alternative<DenseLayer*>(layers[i]))
        {
            //info needed: output, layerType, next layer neurons
            if(holds_alternative<int>(output_size))
            {
                get<DenseLayer*>(layers[i])->init(get<int>(output_size), layerType, i);
            }
            else
            {
                throw runtime_error("The input to a dense layer is not one dimensional");
            }
            cout << "It is a dense layer" << endl;
        }
        else if(holds_alternative<ConvolutionalLayer*>(layers[i]))
        {
            cout << "It is a convolutional layer" << endl;
        }
        else
        {
            cout << "It is a pooling layer" << endl;
        }
    }
}

void NeuralNetwork::stageNetwork(vector<double> input)
{
    for(int i = 0; i < input.size(); i++)
    {
        globalStore.NeuronStore[0][i]->set_value(input[i]);
    }
}

void NeuralNetwork::stageResults()
{
    for(int i = 0; i < this->result_buf.size(); i++)
    {
        this->result_buf[i] = globalStore.NeuronStore[globalStore.NeuronStore.size()-1][i]->get_value();
    }
}

vector<double> NeuralNetwork::forward(vector<double> input)
{
    this->stageNetwork(input);    

    for(int i = 0; i < this->layers.size(); i++)
    {
        if(holds_alternative<DenseLayer*>(layers[i]))
        {
            get<DenseLayer*>(layers[i])->forward();
        }
    }

    this->stageResults();
    return this->result_buf;
}

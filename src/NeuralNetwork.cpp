#include "NeuralNetwork.hpp"
#include <random>
#include <stdexcept>

double learningRate;
double beta1 = 0.9;
double beta2 = 0.999;
int optimizer;
int epoch = 1;
double *neuron_value;
double *cache_value;
double *delta_value;
double *weight;
double *wv;
double *wm;
double *bias;
double *bv;
double *bm;
int *neurons_per_layer;
int *weights_per_layer;
int *biases_per_layer;
int *neuron_acc;
int *weight_acc;
int *bias_acc;

int stringActivationToIntActivation(string activation_function)
{
    if(activation_function == "Linear" || activation_function == "") return 0;
    if(activation_function == "ReLU") return 1;
    if(activation_function == "SoftMax" || activation_function == "Softmax") return 2;
    return 0;
}

int stringInitializationToIntInitialization(string initialization)
{
    if(initialization == "Zero" || initialization == "") return -1;
    if(initialization == "Xavier") return 1;
    if(initialization == "0.5") return 12;
    if(initialization == "HeRandom") return 0;
    return -1;
}

void NeuralNetwork::setGlobals(int opt, double lr)
{
    learningRate = lr;
    optimizer = opt;
}

int stringOptimizerToIntOptimizer(string optimizer)
{
    if(optimizer == "None") return 0;
    if(optimizer == "Adam") return 1;
    return 0;
}


void NeuralNetwork::createResultBuffer()
{
    if(holds_alternative<DenseLayer*>(this->layers[this->layers.size()-1]))
    {
        int outputNodes = get<DenseLayer*>(this->layers[this->layers.size()-1])->getIn();
        this->result_buf = vector<double>(outputNodes, 0);
    }
}

void NeuralNetwork::initialize()
{
    seed = this->seed;
    for(unsigned long int i = 0; i < layers.size(); i++)
    {
        int output_size = 0, act_function = 0;
        if(i != layers.size()-1)
        {
            if(holds_alternative<DenseLayer*>(layers[i+1]))
            {
                output_size = get<DenseLayer*>(layers[i+1])->getIn();
                act_function = get<DenseLayer*>(layers[i+1])->getAct();
            }
            else if(holds_alternative<ConvolutionalLayer*>(layers[i+1]))
            {
                output_size = get<ConvolutionalLayer*>(layers[i+1])->getIn();
                act_function = get<ConvolutionalLayer*>(layers[i+1])->getAct();
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
            get<DenseLayer*>(layers[i])->init(output_size, layerType, i, act_function);
        }
        else if(holds_alternative<ConvolutionalLayer*>(layers[i]))
        {
            get<ConvolutionalLayer*>(layers[i])->init(layerType, i, act_function);
        }
        else
        {
        }
    }
    int total_neurons = 0;
    int total_weights = 0;
    int total_biases = 0;
    for(unsigned long int i = 0; i < this->layers.size(); i++)
    {
        total_neurons += neurons_per_layer[i];
        total_weights += weights_per_layer[i];
        total_biases += biases_per_layer[i];
    }
    neuron_value = new double[total_neurons];
    cache_value = new double[total_neurons];
    delta_value = new double[total_neurons];

    weight = new double[total_weights];
    wv = new double[total_weights];
    wm = new double[total_weights];

    bias = new double[total_biases];
    bv = new double[total_biases];
    bm = new double[total_biases];
    this->final_pass();
    this->createResultBuffer();
}

void NeuralNetwork::final_pass()
{
    for(unsigned long int i = 0; i < this->layers.size(); i++)
    {
        if(holds_alternative<DenseLayer*>(this->layers[i]))
        {
            get<DenseLayer*>(this->layers[i])->init2();
        }
        if(holds_alternative<ConvolutionalLayer*>(this->layers[i]))
        {
            get<ConvolutionalLayer*>(this->layers[i])->init2();
        }
    }
}

void NeuralNetwork::stageNetwork(vector<double> input)
{
    for(unsigned long int i = 0; i < input.size(); i++)
    {
        neuron_value[i] = input[i];
    }
}

void NeuralNetwork::stageResults()
{
    int nStart = neuron_acc[this->layers.size()-1];
    for(unsigned long int i = 0; i < this->result_buf.size(); i++)
    {
        this->result_buf[i] = neuron_value[nStart+i];
    }
}

vector<double> NeuralNetwork::forward(vector<double> input)
{
    this->stageNetwork(input);    

    for(unsigned long int i = 0; i < this->layers.size()-1; i++)
    {
        if(holds_alternative<DenseLayer*>(this->layers[i]))
        {
            get<DenseLayer*>(this->layers[i])->forward();
        }
        if(holds_alternative<ConvolutionalLayer*>(this->layers[i]))
        {
            get<ConvolutionalLayer*>(this->layers[i])->forward();
        }
    }

    this->stageResults();
    return this->result_buf;
}

void NeuralNetwork::backward(vector<double> errors)
{
    if(holds_alternative<DenseLayer*>(this->layers[this->layers.size()-1]))
    {
        get<DenseLayer*>(this->layers[this->layers.size()-1])->firstDeltas(errors);
    }

    for(unsigned long int i = this->layers.size()-2; i > 0; i--)
    {
        if(holds_alternative<DenseLayer*>(this->layers[i]))
        {
            get<DenseLayer*>(this->layers[i])->backward();
        }
    }
}

void NeuralNetwork::update()
{
    for(unsigned long int i = 0; i < this->layers.size()-1; i++)
    {
        if(holds_alternative<DenseLayer*>(this->layers[i]))
        {
            get<DenseLayer*>(this->layers[i])->update();
        }
    }

    epoch++;
}

int NeuralNetwork::get_seed()
{
    return this->seed;
}

void NeuralNetwork::set_seed(int seed)
{
    this->user_set_seed = true;
    this->seed = seed;
}

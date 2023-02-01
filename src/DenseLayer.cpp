#include "NeuralNetwork.hpp"
#include "nnalgorithms.hpp"

DenseLayer::DenseLayer(int nodes, string activation, string initialization, bool bias)
{
    this->in = nodes;
    this->activation = stringActivationToIntActivation(activation);
    this->initialization = stringInitializationToIntInitialization(initialization);
    this->hasBias = bias;

    switch(this->activation)
    {
        case 1:
            this->act_function_derivative = ReLUDerivative;
            break;
        default:
            this->act_function_derivative = LinearDerivative;
            break;
    }
}

int DenseLayer::getIn()
{
    return this->in;
}

int DenseLayer::getAct()
{
    return this->activation;
}

void DenseLayer::init(int output, int layerType, int idx, int next_layer_act)
{
    this->out = output;
    this->layerType = layerType;
    this->idx = idx;
    neurons_per_layer[this->idx] = this->in;
    weights_per_layer[this->idx] = this->in*this->out;
    if(layerType < 2)
        this->hasBias = false;
    switch(next_layer_act)
    {
        case 1:
            this->act_function = ReLU;
            break;
        default:
            this->act_function = Linear;
            break;
    }
    if(this->idx == 0)
    {
        neuron_acc[this->idx] = 0;
        weight_acc[this->idx] = 0;
        bias_acc[this->idx] = 0;

        neuron_acc[this->idx+1] = this->in;
        weight_acc[this->idx+1] = this->in*this->out;
        bias_acc[this->idx+1] = 0;
        biases_per_layer[this->idx] = 0;
    }
    else if(this->layerType != 1)
    {
        neuron_acc[this->idx+1] = this->in + neuron_acc[this->idx];
        weight_acc[this->idx+1] = this->in*this->out + weight_acc[this->idx];
        if(this->hasBias)
        {
            bias_acc[this->idx+1] = 1 + bias_acc[this->idx];
            biases_per_layer[this->idx] = 1;
        }
    }
    else
    {
        biases_per_layer[this->idx] = 0;
    }
}

void DenseLayer::init2()
{
    Initializer init(this->initialization, this->in, this->out);
    int wStart = weight_acc[this->idx];
    for(int i = 0; i < this->in*this->out; i++)
    {
        weight[wStart+i] = init.init();
        wv[wStart+i] = 0;
        wm[wStart+i] = 0;
    }
    if(this->hasBias)
    {
        int bStart = bias_acc[this->idx];
        bias[bStart] = 0;
    }
}

void DenseLayer::forward()
{
    int nStart = neuron_acc[this->idx];
    int wStart = weight_acc[this->idx];
    int bStart = bias_acc[this->idx];
    int nStart_next = neuron_acc[this->idx+1];

    for(int i = 0; i < this->out; i++)
    {
        int jdx = -this->out;
        double sum = 0;
        for(int j = 0; j < this->in; j++)
        {
            jdx += this->out;
            sum += neuron_value[nStart+j]*weight[wStart+jdx+i];
        }
        if(this->hasBias) sum+=bias[bStart];
        neuron_value[nStart_next+i] = this->act_function(sum);
        cache_value[nStart_next+i] = sum;
    }
}

void DenseLayer::firstDeltas(vector<double> errors)
{
    int nStart = neuron_acc[this->idx];
    for(int i = 0; i < this->in; i++)
    {
        delta_value[nStart+i] = (neuron_value[nStart+i] - errors[i]) * this->act_function_derivative(delta_value[nStart+i]);
    }
}

void DenseLayer::backward()
{
    int nStart = neuron_acc[this->idx];
    int wStart = weight_acc[this->idx];
    int nStart_next = neuron_acc[this->idx+1];
    int idx = -this->out;
    for(int i = 0; i < this->in; i++)
    {
        idx += this->out;
        double sum = 0;
        double neuron_derivative = this->act_function_derivative(cache_value[nStart+i]);
        for(int j = 0; j < this->out; j++)
        {
            sum += weight[wStart + j + idx] * delta_value[nStart_next+j] * neuron_derivative;
        }
        delta_value[nStart+i] = sum;
    }
}

void DenseLayer::update()
{
    int nStart = neuron_acc[this->idx];
    int wStart = weight_acc[this->idx];
    int bStart = bias_acc[this->idx];
    int nStart_next = neuron_acc[this->idx+1];
    int idx = -this->out;
    double delta_sum = 0;
    for(int i = 0; i < this->in; i++)
    {
        idx += this->out;
        for(int j = 0; j < this->out; j++)
        {
            double gradient = neuron_value[nStart+i] * delta_value[nStart_next+j];
            delta_sum += delta_value[nStart_next];
            int w = wStart+idx+j;
            wm[w] = beta1 * wm[w] + (1 - beta1) * gradient;
            wv[w] = beta2 * wv[w] + (1 - beta2) * pow(gradient, 2);
            double mhat = wm[w] / (1-pow(beta1, epoch));
            double vhat = wv[w] / (1-pow(beta2, epoch));
            weight[w] -= (learningRate / (sqrt(vhat + 1e-8)) * mhat);
        }
    }
    if(this->hasBias)
    {
        bm[bStart] = beta1 * bm[bStart] + (1 - beta1) * delta_sum;
        bv[bStart] = beta2 * bv[bStart] + (1 - beta2) * pow(delta_sum, 2);
        double mhat = bm[bStart] / (1-pow(beta1, epoch));
        double vhat = bv[bStart] / (1-pow(beta2, epoch));
        bias[bStart] -= (learningRate / (sqrt(vhat + 1e-8)) * mhat);
    }
}




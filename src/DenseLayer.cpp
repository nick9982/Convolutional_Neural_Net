#include "NeuralNetwork.hpp"
#include "nnalgorithms.hpp"
vector<double> resUpdate;
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
        case 2:
            this->softmax_derivative = SoftMaxDerivative;
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
    this->next_activation = next_layer_act;
    neurons_per_layer[this->idx] = this->in;
    weights_per_layer[this->idx] = this->in*this->out;
    if(layerType < 2)
        this->hasBias = false;
    switch(next_layer_act)
    {
        case 1:
            this->act_function = ReLU;
            break;
        case 2:
            this->softmax = SoftMax;
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
    if(this->next_activation == 2)
    {
        this->forward_soft();
        return;
    }
    int nStart = neuron_acc[this->idx];
    int wStart = weight_acc[this->idx];
    int bStart = bias_acc[this->idx];
    int nStart_next = neuron_acc[this->idx+1];
    vector<double> sums;

    for(int i = 0; i < this->out; i++)
    {
        int jdx = wStart-this->out;
        double sum = 0;
        for(int j = 0; j < this->in; j++)
        {
            jdx += this->out;
            sum += neuron_value[nStart+j]*weight[jdx+i];
        }
        if(this->hasBias) sum+=bias[bStart];
        wtsdbg.push_back(sum);
        if(isnan(sum))
        {
            for(int i = 0; i < wtsdbg.size(); i++)
            {
                /* cout << "denseForward3" << wtsdbg[i] << endl; */
            }
            /* exit(0); */
        }
        neuron_value[nStart_next+i] = this->act_function(sum);
        cache_value[nStart_next+i] = sum;
    }
}

void DenseLayer::forward_soft()
{
    int nStart = neuron_acc[this->idx];
    int wStart = weight_acc[this->idx];
    int bStart = bias_acc[this->idx];
    int nStart_next = neuron_acc[this->idx+1];

    double* output_cache = new double[this->out];
    vector<double> sums;

    for(int i = 0; i < this->out; i++)
    {
        int jdx = wStart-this->out;
        double sum = 0;
        for(int j = 0; j < this->in; j++)
        {
            jdx += this->out;
            /* if(i == 0)cout << "eq(wt * neur): " << weight[jdx+i] << " * " << neuron_value[nStart + j] << endl; */
            sum += neuron_value[nStart+j]*weight[jdx+i];
            sums.push_back(weight[jdx+i]);
            /* if(i == 0) cout << sum << endl; */
        }
        if(this->hasBias) sum+=bias[bStart];
        if(isnan(sum))
        {
            for(int i = 0; i < resUpdate.size(); i++)
            {
                /* cout << "denseForward" << resUpdate[i] << endl; */
            }
            /* exit(0); */
        }
        resUpdate.clear();
        cache_value[nStart_next+i] = sum;
        output_cache[i] = sum;
    }
    output_cache = this->softmax(output_cache, this->out);
    for(int i = 0; i < this->out; i++)
    {
        neuron_value[nStart_next+i] = output_cache[i];
    }
    wtsdbg.clear();
}

void DenseLayer::firstDeltas(vector<double> errors)
{
    if(this->activation == 2)
    {
        this->firstDeltas_soft(errors);
        return;
    }
    int nStart = neuron_acc[this->idx];
    for(int i = 0; i < this->in; i++)
    {
        delta_value[nStart+i] = (neuron_value[nStart+i] - errors[i]) * this->act_function_derivative(cache_value[nStart+i]);
    }
}

void DenseLayer::firstDeltas_soft(vector<double> errors)
{
    int nStart = neuron_acc[this->idx];
    double *cache = new double[this->in];
    for(int i = 0; i < this->in; i++)
    {
        cache[i] = cache_value[nStart+i];
        resUpdate.push_back(cache[i]);
    }
    cache = this->softmax_derivative(cache, this->in);//fix before run!

    for(int i = 0; i < this->in; i++)
    {
        delta_value[nStart+i] = (neuron_value[nStart+i] - errors[i]) * cache[i];
        resUpdate.push_back(cache[i]);
    }
}

void DenseLayer::backward()
{
    int nStart = neuron_acc[this->idx];
    int wStart = weight_acc[this->idx];
    int nStart_next = neuron_acc[this->idx+1];
    int idx = wStart-this->out;
    for(int i = 0; i < this->in; i++)
    {
        idx += this->out;
        double sum = 0;
        double neuron_derivative = this->act_function_derivative(cache_value[nStart+i]);
        for(int j = 0; j < this->out; j++)
        {
            sum += weight[j + idx] * delta_value[nStart_next+j] * neuron_derivative;
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
    int idx = wStart-this->out;
    double delta_sum = 0;
    for(int i = 0; i < this->in; i++)
    {
        idx += this->out;
        for(int j = 0; j < this->out; j++)
        {
            double gradient = neuron_value[nStart+i] * delta_value[nStart_next+j];
            delta_sum += delta_value[nStart_next+j];
            int w = idx+j;
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




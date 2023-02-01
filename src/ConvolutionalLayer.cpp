#include "NeuralNetwork.hpp"
#include "nnalgorithms.hpp"

ConvolutionalLayer::ConvolutionalLayer(vector<int> dimensions, vector<int> kernel_dimensions, int new_kernels, string activation, string initialization, vector<int> padding, bool hasBias)
{
    this->x = dimensions[0];
    this->y = dimensions[1];
    this->input_channels = dimensions[2];
    this->kernel_x = kernel_dimensions[0];
    this->kernel_y = kernel_dimensions[1];
    this->stride_x = kernel_dimensions[2];
    this->stride_y = kernel_dimensions[3];
    this->new_kernels = new_kernels;
    this->padding_x = padding[0];
    this->padding_y = padding[1];
    this->hasBias = hasBias;
    this->activation = stringActivationToIntActivation(activation);
    this->initialization = stringInitializationToIntInitialization(initialization);
    this->out_per_wt_x = ceil((this->x + this->padding_x*2 - this->kernel_x)/this->stride_x + 1);
    this->out_per_wt_y = ceil((this->y + this->padding_y*2 - this->kernel_y)/this->stride_y + 1);
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

void ConvolutionalLayer::init(int layerType, int idx, int next_layer_act)
{
    this->layerType = layerType;
    this->idx = idx;
    if(layerType < 2)
    {
        this->hasBias = false;
    }

    neurons_per_layer[this->idx] = this->x * this->y * this->input_channels;
    weights_per_layer[this->idx] = this->kernel_x * this->kernel_y * this->new_kernels * this->input_channels;
    if(this->hasBias) biases_per_layer[this->idx] = this->new_kernels * this->input_channels;
    else biases_per_layer[this->idx] = 0;
    
    switch(next_layer_act)
    {
        case 1:
            this->act_function = ReLU;
            break;
        case 2:
            this->act_function = Linear;
            break;
    }

    if(this->idx == 0)
    {
        neuron_acc[this->idx] = 0;
        weight_acc[this->idx] = 0;
        bias_acc[this->idx] = 0;

        neuron_acc[this->idx+1] = neurons_per_layer[this->idx];
        weight_acc[this->idx+1] = weights_per_layer[this->idx];
        bias_acc[this->idx+1] = 0;
    }
    else if(this->layerType != 1)
    {
        neuron_acc[this->idx+1] = neurons_per_layer[this->idx] + neuron_acc[this->idx];
        weight_acc[this->idx+1] = weights_per_layer[this->idx] + weight_acc[this->idx];
        bias_acc[this->idx+1] = biases_per_layer[this->idx] + bias_acc[this->idx];
    }
}

void ConvolutionalLayer::init2()
{
    int dimension = this->out_per_wt_x * this->out_per_wt_y;
    Initializer init(this->initialization, dimension, dimension);
    int wStart = weight_acc[this->idx];
    for(int i = 0; i < weights_per_layer[this->idx]; i++)
    {
        weight[wStart+i] = init.init();
        wv[wStart+i] = 0;
        wm[wStart+i] = 0;
    }
    if(this->hasBias)
    {
        int bStart = bias_acc[this->idx];
        for(int i = 0; i < biases_per_layer[this->idx]; i++)
        {
            bias[bStart+i] = 0;
        }
    }
}

void ConvolutionalLayer::forward()
{
    int MAXX = this->x-this->padding_x, MAXY = this->y-this->padding_y;
    int nStart = neuron_acc[this->idx];
    int wStart = weight_acc[this->idx];
    int nStart_next = neuron_acc[this->idx];
    int CHAN_SIZE_N = MAXX+MAXY, CHAN_SIZE_W = this->kernel_x*this->kernel_y*this->new_kernels;
    int dim_of_kerns = this->kernel_x*this->kernel_y;

    int chandn = nStart-CHAN_SIZE_N, chandw = wStart-CHAN_SIZE_W;
    //I have come up with idea to store each dimension in a hash map. As it will go
    //through the same dimensions each time, no need to repeat calculations.
    //i.e. no need for the following line.
    int indim1 = this->kernel_x * this->kernel_y * this->new_kernels;
    for(int i = 0; i < this->input_channels; i++)
    {
        chandn += CHAN_SIZE_N;
        chandw += CHAN_SIZE_W;
        int jdx = chandn-MAXY;
        for(int j = this->padding_x; j < MAXX; j+=this->stride_x)
        {
            jdx += MAXY;
            for(int k = this->padding_y; k < MAXY; k+=this->stride_y)
            {
                neuron_value[jdx+k];
                int wdx = chandw-dim_of_kerns;
                //This layer does iterate through each neuron of each channel row wise
                for(int w = 0; w < this->new_kernels; w++)
                {
                    wdx += dim_of_kerns;
                    double sum = 0;
                    int xdx = wdx-this->kernel_y;
                    for(int x = 0; x < this->kernel_x; x++)
                    {
                        xdx += this->kernel_y;
                        if(x+j < MAXX)
                        {
                            for(int y = 0; y < this->kernel_y; y++)
                            {
                                if(y+k < MAXY)
                                {
                                    sum += neuron_value[jdx+k] * weight[xdx + y];

                                    /* sum += globalStore.NeuronStore[this->idx][neur++]->get_value() * */
                                    /*     globalStore.WeightStore[this->idx][]; */
                                }
                            }
                            
                        }
                    }
                    //TODO: need to locate this neuron
                    //the output neuron
                }
            }
        }
    }
}

int ConvolutionalLayer::getIn()
{
    return this->x*this->y*this->input_channels;
}

int ConvolutionalLayer::getAct()
{
    return this->activation;
}

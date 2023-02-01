#include "NeuralNetwork.hpp"

ConvolutionalLayer::ConvolutionalLayer(vector<int> dimensions, vector<int> kernel_dimensions, int new_kernels, string activation, string initialization, vector<int> padding)
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
    this->activation = stringActivationToIntActivation(activation);
    this->initialization = stringInitializationToIntInitialization(initialization);
    this->out_per_wt_x = ceil((this->x + this->padding_x*2 - this->kernel_x)/this->stride_x + 1);
    this->out_per_wt_y = ceil((this->y + this->padding_y*2 - this->kernel_y)/this->stride_y + 1);
}

void ConvolutionalLayer::init(int layerType, int idx)
{
    this->layerType = layerType;
    this->idx = idx;
    
    /* globalStore.createConvLayer(this->x, this->y, this->input_channels, this->kernel_x, this->kernel_y, this->new_kernels, this->activation, this->initialization, this->out_per_wt_x * this->out_per_wt_y); */
    if(this->layerType > 1)
    {
        /* globalStore.BiasStore[this->idx][0]->set_exists(true); */
    }
}

void ConvolutionalLayer::forward()
{
    int MAXX = this->x-this->padding_x, MAXY = this->y-this->padding_y;
    int neur = 0;
    int wt = 0;
    //I have come up with idea to store each dimension in a hash map. As it will go
    //through the same dimensions each time, no need to repeat calculations.
    //i.e. no need for the following line.
    int indim1 = this->kernel_x * this->kernel_y * this->new_kernels;
    for(int i = 0; i < this->input_channels; i++)
    {
        wt += i * indim1;
        for(int j = this->padding_x; j < MAXX; j+=this->stride_x)
        {
            for(int k = this->padding_y; k < MAXY; k+=this->stride_y)
            {
                //This layer does iterate through each neuron of each channel row wise
                double sum = 0;
                for(int x = 0; x < this->kernel_x; x++)
                {
                    if(x+j < MAXX)
                    {
                        for(int y = 0; y < this->kernel_y; y++)
                        {
                            if(y+k < MAXY)
                            {
                                /* sum += globalStore.NeuronStore[this->idx][neur++]->get_value() * */
                                /*     globalStore.WeightStore[this->idx][]; */
                            }
                        }
                        
                    }
                }
            }
        }
    }
}

vector<int> ConvolutionalLayer::getIn()
{
    vector<int> inputs{this->x, this->y, this->input_channels};
    return inputs;
}

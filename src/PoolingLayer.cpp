#include "NeuralNetwork.hpp"
#include "nnalgorithms.hpp"

int stringPoolTypeToIntPoolType(string pool_type)
{
    if(pool_type == "Max") return 0;
    else if(pool_type == "Average") return 1;
    return 0;//default to Max
}

PoolingLayer::PoolingLayer(vector<int> dimensions, vector<int> kernel_dimensions, string pool_type)
{
    this->x = dimensions[0];
    this->y = dimensions[1];
    this->input_channels = dimensions[2];
    this->kernel_x = kernel_dimensions[0];
    this->kernel_y = kernel_dimensions[1];
    this->stride_x = kernel_dimensions[2];
    this->stride_y = kernel_dimensions[3];
    this->poolType = stringPoolTypeToIntPoolType(pool_type);
    this->out_x = ceil((double)(this->x - this->kernel_x)/(double)this->stride_x + 1);
    this->out_y = ceil((double)(this->y - this->kernel_y)/(double)this->stride_y + 1);
    this->idx_cache = new int[this->out_x*this->out_y];
}

void PoolingLayer::init(int layerType, int idx, int next_layer_act)
{
    this->layerType = layerType;
    this->idx = idx;
    neurons_per_layer[this->idx] = this->x*this->y*this->input_channels;

    switch(next_layer_act)
    {
        case 1:
            this->act_function = ReLU;
            break;
        default:
            this->act_function = Linear;
            break;
    }

    neuron_acc[this->idx+1] = neurons_per_layer[this->idx] + neuron_acc[this->idx];
    weight_acc[this->idx] = weight_acc[this->idx-1];
    bias_acc[this->idx] = bias_acc[this->idx-1];
}

void PoolingLayer::forward()
{
    int nStart = neuron_acc[this->idx];
    int nStart_next = neuron_acc[this->idx+1];
    int CHAN_SIZE_N = this->x*this->y;
    int next_layer_size = this->out_x*this->out_y;
    int chandn = nStart - CHAN_SIZE_N;
    int mulp = this->y * this->stride_x;

    for(int i = 0; i < this->input_channels; i++)
    {
        chandn += CHAN_SIZE_N;
        int jdx = chandn - mulp;
        for(int j = 0; j < this->x; j+=stride_x)
        {
            jdx += mulp;
            for(int k = 0; k < this->y; k+=stride_y)
            {
                int adj_neur_acc = jdx - this->y;
                int idx = jdx + k;
                int index = nStart_next + i * next_layer_size + j/this->stride_x * this->out_y + k/this->stride_y;
                if(this->poolType == 0)
                {
                    double MAX = -4293918720;
                    for(int r = 0; r < this->kernel_x; r++)
                    {
                        if(r+j >= this->x) break;
                        adj_neur_acc += this->y;
                        for(int w = 0; w < this->kernel_y; w++)
                        {
                            if(w+k >= this->y) break;
                            if(neuron_value[adj_neur_acc+k+w] > MAX)
                            {
                                MAX = neuron_value[adj_neur_acc+k+w];
                                idx_cache[index] = adj_neur_acc+k+w;
                            }
                        }
                    }
                    cache_value[index] = MAX;
                    neuron_value[index] = this->act_function(MAX);
                }
                else
                {
                    double sum = 0;
                    int count = 0;
                    for(int r = 0; r < this->kernel_x; r++)
                    {
                        if(r+j >= this->x) break;
                        adj_neur_acc += this->y;
                        for(int w = 0; w < this->kernel_y; w++)
                        {
                            if(w+k >= this->y) break;
                            sum += neuron_value[adj_neur_acc+k+w];
                            count++;
                        }
                    }
                    this->idx_cache[idx] = count;
                    double avg = sum/(double)count;
                    cache_value[index] = avg;
                    neuron_value[index] = this->act_function(avg);
                }
            }
        }
    }
}

void PoolingLayer::backward()
{
    int nStart = neuron_acc[this->idx];
    int nStart_next = neuron_acc[this->idx+1];
    if(this->poolType == 0)
    {
        for(int i = 0; i < neurons_per_layer[this->idx+1]; i++)
        {
            delta_value[nStart + this->idx_cache[i]] = delta_value[nStart_next+i];
        }
    }
    else
    {
        //not sure if this is how it works.
        
    }
}

int PoolingLayer::getIn()
{
    return this->x * this->y * this->input_channels;
}

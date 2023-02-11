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
    this->out_per_wt_x = ceil((double)(this->x + this->padding_x*2 - this->kernel_x)/(double)this->stride_x + 1);
    this->out_per_wt_y = ceil((double)(this->y + this->padding_y*2 - this->kernel_y)/(double)this->stride_y + 1);
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
        default:
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
vector<string> links;

void ConvolutionalLayer::forward()
{
    int nStart = neuron_acc[this->idx];
    int wStart = weight_acc[this->idx];
    int nStart_next = neuron_acc[this->idx+1];
    int CHAN_SIZE_N = this->x*this->y, CHAN_SIZE_W = this->kernel_x*this->kernel_y*this->new_kernels;
    int dim_of_kerns = this->kernel_x*this->kernel_y;
    int biasCnt = bias_acc[this->idx], out = 0;
    int next_lay_size = this->out_per_wt_x * this->out_per_wt_y;
    int mulp = this->y*this->stride_x;
    int chandn = nStart - CHAN_SIZE_N, chandw = wStart - CHAN_SIZE_W;
    vector<double> sums;

    for(int i = 0; i < this->input_channels; i++)
    {
        chandn += CHAN_SIZE_N;
        chandw += CHAN_SIZE_W;
        int jdx = chandw - dim_of_kerns;
        for(int j = 0; j < this->new_kernels; j++)
        {
            jdx += dim_of_kerns;
            int kdx = chandn - mulp;
            //JUST ADDED THIS CODE FOR +THIS-STRIDE_X-1 THAT COULD BREAK THE PROGRAM DK
            for(int k = 0; k < this->x+this->stride_x; k+=this->stride_x)
            {
                kdx += mulp;
                for(int l = 0; l < this->y+this->stride_y; l+=this->stride_y)
                {
                    double sum = 0;
                    int adj_neur_acc = kdx - this->y;
                    int idx = nStart_next + out * next_lay_size + k/this->stride_x * this->out_per_wt_y + l/this->stride_y;
                    int rdx = jdx - this->kernel_y;
                    for(int r = 0; r < this->kernel_x; r++)
                    {
                        if(r+k >= this->x) break;
                        rdx += this->kernel_y;
                        adj_neur_acc += this->y;
                        for(int w = 0; w  < this->kernel_y; w++)
                        {
                            if(w+l >= this->y) break;
                            sum += neuron_value[adj_neur_acc+l+w] * weight[rdx+w];
                            /* if(this->idx  == 1) cout << adj_neur_acc << endl; */
                            /* if(this->idx == 0) cout << adj_neur_acc+l+w << endl; */
                            if(this->idx == 1)
                            {
                                string link = to_string(adj_neur_acc+l+w) + ", " + to_string(rdx+w) + ", " + to_string(idx);
                                /* string link = to_string(adj_neur_acc+l+w) + ", " + to_string(rdx+w); */
                                /* string link = to_string(kdx+l) + ", " + to_string(idx); */
                                links.push_back(link);
                            }
                        }
                    }
                    if(this->hasBias)
                    {
                        sum += bias[biasCnt];
                    }
                    /* cout << sum << endl; */
                    wtsdbg.push_back(sum);
                    if(isnan(sum))
                    {
                        exit(0);
                    }
                    cache_value[idx] = sum;
                    neuron_value[idx] = this->act_function(sum);
                }
            }
            biasCnt++;
            out++;
        }
    }
}

vector<string> linksback;
void check_vectors();
/* void ConvolutionalLayer::backward() */
/* { */
/*     int MAXX = this->x-this->padding_x, MAXY = this->y-this->padding_y; */
/*     int nStart = neuron_acc[this->idx]; */
/*     int wStart = weight_acc[this->idx]; */
/*     int bStart = bias_acc[this->idx]; */
/*     int nStart_next = neuron_acc[this->idx+1]; */
/*     int CHAN_SIZE_N = MAXX*MAXY, CHAN_SIZE_W = this->kernel_x*this->kernel_y*this->new_kernels; */
/*     int dim_of_kerns = this->kernel_x*this->kernel_y; */
/*     int next_lay_size = this->out_per_wt_x * this->out_per_wt_y; */
/*      */
/*     int chandn = nStart - CHAN_SIZE_N, chandw = wStart - CHAN_SIZE_W; */
/*     for(int i = 0; i < this->input_channels; i++) */
/*     { */
/*         chandn += CHAN_SIZE_N; */
/*         chandw += CHAN_SIZE_W; */
/*         int jdx = chandn - MAXY; */
/*         for(int j = 0; j < MAXX; j++) */
/*         { */
/*             int OFFS_X = floor((double)(j%this->kernel_x) / this->stride_x); */
/*             jdx += MAXY; */
/*             for(int k = 0; k < MAXY; k++) */
/*             { */
/*                 int OFFS_Y = floor((double)(k%this->kernel_y) / this->stride_y); */
/*                 double neuron_derivative = this->act_function_derivative(cache_value[jdx+k]); */
/*                 double sum = 0; */
/*                 int rdx = chandw - dim_of_kerns; */
/*                 for(int r = 0; r < this->new_kernels; r++) */
/*                 { */
/*                     int idx = nStart_next + ((i*this->new_kernels)+r) * next_lay_size + (ceil((double)j/this->stride_x)+OFFS_X) * this->out_per_wt_y + ceil((double)k/this->stride_y)+OFFS_Y; */
/*                     rdx += dim_of_kerns; */
/*                     int dispx = 0; */
/*                     for(int x = j%this->kernel_x; x >= 0; x-=this->stride_x) */
/*                     { */
/*                         int dispy = 0; */
/*                         for(int y = k%this->kernel_y; y >= 0; y-=this->stride_y) */
/*                         { */
/*                             sum += delta_value[idx - dispx * this->out_per_wt_y - dispy++] * weight[rdx + x*this->kernel_y+y] * neuron_derivative; */
/*                             if(this->idx == 1) */
/*                             { */
/*                                 string link =  to_string(jdx+k) + ", " + to_string(rdx + x*this->kernel_y+y) + ", " + to_string(idx - dispx * this->out_per_wt_y - dispy-1); */
/*                                 linksback.push_back(link); */
/*                             } */
/*                         } */
/*                         dispx++; */
/*                     } */
/*                 } */
/*                 delta_value[jdx+k] = sum; */
/*             } */
/*         } */
/*     } */
/* } */

void ConvolutionalLayer::backward()
{
    int MAXX = this->x-this->padding_x, MAXY = this->y-this->padding_y;
    int nStart = neuron_acc[this->idx];
    int wStart = weight_acc[this->idx];
    int bStart = bias_acc[this->idx];
    int nStart_next = neuron_acc[this->idx+1];
    int CHAN_SIZE_N = MAXX*MAXY, CHAN_SIZE_W = this->kernel_x*this->kernel_y*this->new_kernels;
    int dim_of_kerns = this->kernel_x*this->kernel_y;
    int out_img_dim = this->out_per_wt_x*this->out_per_wt_y;
    int img_dim = this->x*this->y;
    int next_lay_size = this->out_per_wt_x * this->out_per_wt_y;
    int outchan_size = next_lay_size * this->new_kernels;
    
    int chandn = nStart - CHAN_SIZE_N, chandw = wStart - CHAN_SIZE_W;
    int ndx = nStart - this->y;
    int out = nStart_next - outchan_size;
    int weight_y_size = this->kernel_y*this->stride_x;
    for(int k = 0; k < this->input_channels; k++)
    {
        chandn += CHAN_SIZE_N;
        chandw += CHAN_SIZE_W;
        int idx = chandn - this->y;
        out += outchan_size;
        for(int i = 0; i < this->x; i++)
        {
            idx += this->y;
            for(int j = 0; j < this->y; j++)
            {
                double neuron_derivative = this->act_function_derivative(cache_value[idx+j]);
                int base_output_idx = floor((double)i/(double)this->stride_x) * this->out_per_wt_x + floor((double)j/(double)this->stride_y);
                double sum = 0;
                /* if(this->idx == 1) cout << idx+j << endl; */
                int mdx = base_output_idx + out - next_lay_size;
                int mdx_w = chandw - dim_of_kerns;
                for(int m = 0; m < this->new_kernels; m++)
                {
                    mdx_w += dim_of_kerns;
                    mdx += next_lay_size;
                    /* if(this->idx == 1) cout << mdx << endl; */
                    int dispx = mdx + this->out_per_wt_y;
                    int xdx_w = mdx_w + weight_y_size;
                    for(int x = i%this->kernel_x; x >= 0; x-=this->stride_x)
                    {
                        xdx_w -= weight_y_size;
                        dispx -= this->out_per_wt_y;
                        for(int y = j%this->kernel_y; y >= 0; y-=this->stride_y)
                        {
                            sum += delta_value[dispx-y] * weight[xdx_w-y];
                            /* if(this->idx == 1) cout << xdx_w - y << endl; */
                            if(this->idx == 1)
                            {
                                string link = to_string(idx+j) + ", " + to_string(xdx_w-y) + ", " + to_string(dispx-y);
                                /* string link = to_string(idx+j) + ", " + to_string(xdx_w-y); */
                                /* string link = to_string(idx+j) + ", " + to_string(mdx); */
                                linksback.push_back(link);
                            }
                            
                        }
                    }
                    //this point we are at the level of weights
                    /* if(this->idx == 1) cout << kdx << endl; */
                    //at the end of kernel line add size of output once.
                }
            }
        }
    }
    if(this->idx==1)cout << neuron_acc[this->idx] << ", " << neuron_acc[this->idx+1] << ", " << neuron_acc[this->idx+2] << endl;
}

//There are multiple inp neuron X output neuron combinations for each weight.
//The gradient for each weight is the sum of the gradient of these combinations.
//What we really are doing here is cycling through each weight and idxing each
//respective pair of input neuron X output neuron.
//i.e. inp neur = neuron_value[nStart + a] * delta_value[nStart_next + x];
//sum those up for each weight then update.
vector<string> linksupdate;
void ConvolutionalLayer::update()
{
    int MAXX = this->x-this->padding_x, MAXY = this->y-this->padding_y;
    int nStart = neuron_acc[this->idx];
    int wStart = weight_acc[this->idx];
    int biasCnt = bias_acc[this->idx];
    int nStart_next = neuron_acc[this->idx+1];
    int mulp = this->y * this->stride_x;
    int CHAN_SIZE_N = MAXX*MAXY, CHAN_SIZE_W = this->kernel_x*this->kernel_y*this->new_kernels;
    int next_lay_size = this->out_per_wt_x * this->out_per_wt_y;
    int out = 0;

    int chandw = wStart - CHAN_SIZE_W, chandn = nStart - CHAN_SIZE_N;
    int dim_of_kerns = this->kernel_x*this->kernel_y;
    for(int i = 0; i < this->input_channels; i++)
    {
        chandw += CHAN_SIZE_W;
        chandn += CHAN_SIZE_N;
        int jdx = chandw - dim_of_kerns;
        for(int j = 0; j < this->new_kernels; j++)
        {
            jdx += dim_of_kerns;
            int kdx = jdx - this->kernel_y;
            double delta_sum = 0;
            for(int k = 0; k < this->kernel_x; k++)
            {
                kdx += this->kernel_y;
                for(int m = 0; m < this->kernel_y; m++)
                {
                    double weightval = weight[kdx + m];
                    double gradient = 0;
                    //NOW WE ACCESS EACH WEIGHT WITH weight[kdx+m]
                    int xdx = chandn - mulp;
                    int out_x = 0;
                    for(int x = k; x < this->x; x+=this->stride_x)
                    {
                        xdx += mulp;
                        int out_y = 0;
                        for(int y = m; y < this->y; y+=this->stride_y)
                        {
                            int idx = nStart_next + out * next_lay_size + out_x*this->out_per_wt_y + out_y++;
                            gradient += neuron_value[xdx + y] * delta_value[idx];
                            /* if(this->idx == 0)cout << neuron_value[xdx+y] << endl; */
                            if(this->idx==1)
                            {
                                delta_sum += delta_value[idx];
                                string link = to_string(xdx+y) + ", " + to_string(kdx+m) + ", " + to_string(idx);
                                linksupdate.push_back(link);
                            }
                        }
                        out_x++;
                    }
                    int w = kdx+m;
                    wm[w] = beta1 * wm[w] + (1 - beta1) * gradient;
                    wv[w] = beta2 * wv[w] + (1 - beta2) * pow(gradient, 2);
                    double mhat = wm[w] / (1-pow(beta1, epoch));
                    double vhat = wv[w] / (1-pow(beta2, epoch));
                    weight[w] -= (learningRate / (sqrt(vhat + 1e-8)) * mhat);
                    /* cout << "weigth: " << weight[w] << endl; */
                }
            }
            if(this->hasBias)
            {
                int b = biasCnt++;
                bm[b] = beta1 * bm[b] + (1 - beta1) * delta_sum;
                bv[b] = beta2 * bv[b] + (1 - beta2) * pow(delta_sum, 2);
                double mhat = bm[b] / (1-pow(beta1, epoch));
                double vhat = bv[b] / (1-pow(beta2, epoch));
                bias[b] -= (learningRate / (sqrt(vhat + 1e-8)) * mhat);
            }
            out++;
        }
    }
    if(this->idx==1)check_vectors();
}

void check_vectors()
{
    int total_correct = 0;
    for(int i = 0; i < links.size(); i++)
    {
        for(int j = 0; j < linksback.size(); j++)
        {
            if(links[i] == linksback[j])
            {
                total_correct++;
                break;
            }
        }
    }
    double avg = (double)total_correct/(double)links.size() * 100;
    cout << "backward acc: " << to_string(avg) << "%" << endl;
    
    total_correct = 0;
    for(int i = 0; i < links.size(); i++)
    {
        for(int j = 0; j < linksupdate.size(); j++)
        {
            if(links[i] == linksupdate[j])
            {
                total_correct++;
                break;
            }
        }
    }
    avg = (double)total_correct/(double)links.size() * 100;
    cout << "update acc: " << to_string(avg) << "%" << endl;

    total_correct = 0;
    for(int i = 0; i < linksback.size(); i++)
    {
        for(int j = 0; j < linksupdate.size(); j++)
        {
            if(linksupdate[i] == linksback[j])
            {
                total_correct++;
                break;
            }
        }
    }
    avg = (double)total_correct/(double)links.size() * 100;
    cout << "back/update acc: " << to_string(avg) << "%" << endl;

    cout << "front size: " << links.size() << endl;
    cout << "back size: " << linksback.size() << endl;
    cout << "update size: " << linksupdate.size() << endl;
}

int ConvolutionalLayer::getIn()
{
    return this->x*this->y*this->input_channels;
}

int ConvolutionalLayer::getAct()
{
    return this->activation;
}

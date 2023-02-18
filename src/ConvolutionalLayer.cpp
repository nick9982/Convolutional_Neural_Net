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
    int CHAN_SIZE_N = this->x*this->y, CHAN_SIZE_W = this->new_kernels*this->kernel_x*this->kernel_y;
    int kdx_inc = this->y * this->stride_x;
    int chandn = nStart - CHAN_SIZE_N, chandw = wStart-CHAN_SIZE_W;
    int outdim = this->out_per_wt_x*this->out_per_wt_y;
    int out = nStart_next;
    int biasCnt = bias_acc[this->idx];
    int wt_dim = this->kernel_x*this->kernel_y;
    /* int op_cnt = 0; */
    for(int i = 0; i < this->input_channels; i++)
    {
        chandn += CHAN_SIZE_N;
        chandw += CHAN_SIZE_W;
        int jdx = chandw - wt_dim;
        for(int j = 0; j < this->new_kernels; j++)
        {
            jdx += wt_dim;
            int kdx = chandn - kdx_inc;
            for(int k = 0; k < this->x; k+=stride_x)
            {
                kdx += kdx_inc;
                for(int m = 0; m < this->y; m+=stride_y)
                {
                    double sum = 0;
                    int offsx = kdx + m - this->y;
                    int xdx = jdx - this->kernel_y;
                    for(int x = 0; x < this->kernel_x; x++)
                    {
                        if(x+k >= this->x) break;
                        offsx += this->y;
                        xdx += this->kernel_y;
                        for(int y = 0; y < this->kernel_y; y++)
                        {
                            if(y+m >= this->y) break;
                            sum += neuron_value[offsx+y] * weight[xdx+y];
                            /* op_cnt++; */
                            /* if(this->idx == 1) */
                            /* { */
                                /* string link = to_string(offsx+y) + ", " + to_string(xdx+y) + ", " + to_string(out); */
                                /* string link = to_string(offsx+y) + ", " + to_string(xdx+y); */
                                /* links.push_back(link); */
                            /* } */
                        }
                    }
                    if(this->hasBias) sum += bias[biasCnt];
                    neuron_value[out] = this->act_function(sum);
                    /* if(this->idx == 1)cout << "out: " << out << endl; */
                    cache_value[out++] = sum;
                }
            }
            ++biasCnt;
        }
    }
    /* if(this->idx == 1)cout << "cnt_fowrard: " << op_cnt << endl; */
}

vector<string> linksback;
void check_vectors();
void ConvolutionalLayer::backward()
{
    int nStart = neuron_acc[this->idx];
    int wStart = weight_acc[this->idx];
    int CHAN_SIZE_N = this->x*this->y, CHAN_SIZE_W = this->kernel_x*this->kernel_y*this->new_kernels;
    int chandn = nStart - CHAN_SIZE_N, chandw = wStart - CHAN_SIZE_W;
    int OUT_SIZE = this->out_per_wt_x*this->out_per_wt_x;
    int outchan_relative_size = OUT_SIZE * this->new_kernels;
    int out = neuron_acc[this->idx+1] - outchan_relative_size;
    int krnl_size = this->kernel_x*this->kernel_y;
    int mulp = this->kernel_y*this->stride_x;
    /* int cnt = 0; */
    for(int i = 0; i < this->input_channels; i++)
    {
        chandn += CHAN_SIZE_N;
        chandw += CHAN_SIZE_W;
        out += outchan_relative_size;
        int jdx = chandn - this->y;
        int rel_out = -1;
        int str_x = -this->out_per_wt_y;
        for(int j = 0; j < this->x; j++)
        {
            int x_mod = j%this->stride_x;
            int wt_start = x_mod*this->kernel_y;
            if(x_mod == 0) str_x += this->out_per_wt_y;
            jdx += this->y;
            int str_y = str_x-1;
            for(int k = 0; k < this->y; k++)
            {
                int y_mod = k%this->stride_y;
                if(y_mod == 0) str_y++;
                int mdx = out - OUT_SIZE;
                double neuron_derivative = this->act_function_derivative(cache_value[jdx+k]);
                double sum = 0;
                int mdx_w = chandw-krnl_size;
                /* if(this->idx == 1) cout << "neur: " << str_y << endl; */
                for(int m = 0; m < this->new_kernels; m++)
                {
                    mdx_w += krnl_size;
                    mdx += OUT_SIZE;
                    /* if(this->idx == 1) cout << "mdx: " << mdx << endl; */
                    /* if(this->idx == 1)cout << "mdx: " << mdx << endl; */
                    int xdx = str_y + this->out_per_wt_y;
                    int xdx_w = mdx_w + wt_start - mulp;
                    for(int x = x_mod; x < this->kernel_x; x+=this->stride_x)
                    {
                        if(j-x < 0) break;
                        xdx_w += mulp;
                        xdx -= this->out_per_wt_y;
                        int ydx = xdx;
                        for(int y = y_mod; y < this->kernel_y; y+=this->stride_y)
                        {
                            if(k-y < 0) break;
                            /* if(this->idx == 1 && j == 0 && k == 0 && i == 0) cout << "wt: " << xdx_w + y << endl; */
                            /* if(this->idx == 1)cout << "ph" << endl; */
                            /* if(this->idx == 1 && xdx_w+y == 36) cout <<"neuron: " << jdx+k << endl; */
                            sum += delta_value[mdx+ydx--] * weight[xdx_w+y] * neuron_derivative;
                            /* if(this->idx == 1) cout << "str_y: " << xdx_w+y << ", x:" << j << ", y: " << k << endl; */
                            /* if(this->idx == 1) */
                            /* { */
                                /* string link = to_string(jdx+k) + ", " + to_string(xdx_w+y) + ", " + to_string(mdx+ydx+1); */
                                /* string link = to_string(jdx+k) + ", " + to_string(xdx_w+y); */
                                /* linksback.push_back(link); */
                            /* } */

                            /* cnt++; */
                        }
                    }
                }
                delta_value[jdx+k] = sum;
            }
        }
    }
    /* if(this->idx == 1) cout << weight_acc[this->idx] << ", " << weight_acc[this->idx+1] << ", " << weight_acc[this->idx+2] << endl;  */
    /* if(this->idx == 1) cout << neuron_acc[this->idx] << ", " << neuron_acc[this->idx+1] << ", " << neuron_acc[this->idx+2] << endl;  */
    /* if(this->idx == 1) cout << "cnt: " << cnt << endl; */
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
    int nStart = neuron_acc[this->idx];
    int wStart = weight_acc[this->idx];
    int biasCnt = bias_acc[this->idx];
    int CHAN_SIZE_N = this->x*this->y, CHAN_SIZE_W = this->new_kernels*this->kernel_x*this->kernel_y, K_SIZE = this->kernel_x*this->kernel_y;
    int mulp = this->y*this->stride_x;
    int outn_size = this->out_per_wt_x*this->out_per_wt_y;
    int chandn = nStart - CHAN_SIZE_N, chandw = wStart-CHAN_SIZE_W, outn = neuron_acc[this->idx+1];
    int cnt = 0;
    for(int i = 0; i < this->input_channels; i++)
    {
        chandw += CHAN_SIZE_W;
        chandn += CHAN_SIZE_N;
        int jdx = chandw - K_SIZE;
        for(int j = 0; j < this->new_kernels; j++)
        {
            double delta_sum = 0;
            jdx += K_SIZE;
            int kdx = jdx - this->kernel_y;
            for(int k = 0; k < this->kernel_x; k++)
            {
                kdx += this->kernel_y;
                int neur_disp = k*this->y;
                for(int m = 0; m < this->kernel_y; m++)
                {
                    double gradient = 0;
                    //weight: kdx+m
                    /* if(this->idx==1)cout << "weight: " << kdx+m << endl; */
                    int xdx = chandn - mulp + neur_disp;
                    int xdx_out = outn-this->out_per_wt_y;
                    for(int x = k; x < this->x; x+=this->stride_x)
                    {
                        xdx_out += this->out_per_wt_y;
                        xdx += mulp;
                        int ydx_out = xdx_out;
                        for(int y = m; y < this->y; y+=this->stride_y)
                        {
                            //Neuron: xdx+y
                            /* if(this->idx == 1)cout << "mulp: " << xdx+y << endl; */
                            delta_sum += delta_value[ydx_out];
                            gradient += delta_value[ydx_out++] * neuron_value[xdx+y];
                            /* if(this->idx == 1) */
                            /* { */
                            /*     string link = to_string(xdx+y) + ", " + to_string(kdx+m) + ", " + to_string(ydx_out-1); */
                            /*     linksupdate.push_back(link); */
                            /* } */
                            cnt++;
                        }
                    }
                    int w = kdx+m;
                    wm[w] = beta1 * wm[w] + (1 - beta1) * gradient;
                    wv[w] = beta2 * wv[w] + (1 - beta2) * pow(gradient, 2);
                    double mhat = wm[w] / (1-pow(beta1, epoch));
                    double vhat = wv[w] / (1-pow(beta2, epoch));
                    weight[w] -= learningRate * gradient;
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
            outn+=outn_size;
        }
    }

    /* if(this->idx == 1) cout << "cnt: " << cnt << endl; */
    /* if(this->idx==1)cout << neuron_acc[this->idx] << ", " << neuron_acc[this->idx+1] << ", " << neuron_acc[this->idx+2] << endl; */
    /* if(this->idx==1)cout << weight_acc[this->idx] << ", " << weight_acc[this->idx+1] << ", " << weight_acc[this->idx+2] << endl; */
    /* if(this->idx==1)check_vectors(); */
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

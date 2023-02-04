#include <stdarg.h>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#include <variant>
#include <string>
#include <cstdint>
#include <unordered_map>

#define noop (void)0

using namespace std;
extern double *neuron_value;
extern double *cache_value;
extern double *delta_value;
extern double *weight;
extern double *wv;
extern double *wm;
extern double *bias;
extern double *bv;
extern double *bm;
extern int *neurons_per_layer;
extern int *weights_per_layer;
extern int *biases_per_layer;
extern int *neuron_acc;
extern int *weight_acc;
extern int *bias_acc;

class Initializer
{
    public:
        Initializer(int, int, int);
        double init();
    private:
        int init_formula, in, out;
};




int stringActivationToIntActivation(string);
int stringInitializationToIntInitialization(string);
int randomize();

class DenseLayer
{
    private:
        int in, out, initialization, activation, layerType, idx, next_activation;
        bool hasBias;
        double (*act_function)(double);
        double (*act_function_derivative)(double);
        double* (*softmax)(double*, int);
        double* (*softmax_derivative)(double*, int);
    public:
        DenseLayer(int, string, string, bool=true);
        void init(int, int, int, int);
        void init2();
        void forward();
        void forward_soft();
        void backward();
        void firstDeltas(vector<double>);
        void update();
        int getIn();
        int getAct();
};

class ConvolutionalLayer
{
    private:
        int x, y, input_channels, kernel_x, kernel_y, stride_x, stride_y, new_kernels, activation, initialization, layerType, idx, padding_x, padding_y, out_per_wt_x, out_per_wt_y;
        bool hasBias;
        double (*act_function)(double);
        double (*act_function_derivative)(double);
    public:
        ConvolutionalLayer(vector<int>, vector<int>, int, string, string, vector<int> = {0, 0}, bool = true);
        void init(int, int, int);
        void init2();
        void forward();
        int getIn();
        int getAct();
        void backward();
        void update();
};

class PoolingLayer
{
    private:
    public:
        PoolingLayer();
};

class Flatten
{
    private:
    public:
};

int stringOptimizerToIntOptimizer(string);

typedef variant<string, double, int, DenseLayer*, ConvolutionalLayer*, PoolingLayer*> Cover;
typedef variant<DenseLayer*, ConvolutionalLayer*, PoolingLayer*> Layer;

class NeuralNetwork
{
    public:
        //Parameters syntax(string optimizer, double learningRate, ...layers);
        template<typename... Args> NeuralNetwork(Args... args)
        {
            this->seed = randomize();
            const int size = sizeof...(args);
            Cover params[size] = {args...};
            int optimizer = 0;
            bool learningRateSet = false;
            double learningRate = 0.001;
            int layerCnt = 0;
            for(int i = 0; i < sizeof params / sizeof params[0]; i++)
            {
                if(holds_alternative<string>(params[i]) && i == 0)
                {
                    optimizer = stringOptimizerToIntOptimizer(get<string>(params[i]));
                }
                else if(holds_alternative<double>(params[i]) && (i == 0 || i == 1) && !learningRateSet)
                {
                    learningRate = get<double>(params[i]);
                }
                else if(holds_alternative<DenseLayer*>(params[i]))
                {
                    Layer layer = get<DenseLayer*>(params[i]);
                    layers.push_back(layer);
                    layerCnt++;
                }
                else if(holds_alternative<ConvolutionalLayer*>(params[i]))
                {
                    Layer layer = get<ConvolutionalLayer*>(params[i]);
                    layers.push_back(layer);
                    layerCnt++;
                }
                else if(holds_alternative<PoolingLayer*>(params[i]))
                {
                    Layer layer = get<PoolingLayer*>(params[i]);
                    layers.push_back(layer);
                    layerCnt++;
                }
                else if(holds_alternative<int>(params[i]) && i == (sizeof params / sizeof params[0])-1)
                {
                    this->seed = get<int>(params[i]);
                }
                else
                {
                    throw runtime_error("The parameters are not properly set. Format reminder (string optimizer(optional), double learningRate(optional), ...layers, int seed(optional))");
                }
            }
            neurons_per_layer = new int[layerCnt];
            weights_per_layer = new int[layerCnt];
            biases_per_layer = new int[layerCnt];
            neuron_acc = new int[layerCnt];
            weight_acc = new int[layerCnt];
            bias_acc = new int[layerCnt];
            srand(this->seed);
            setGlobals(optimizer, learningRate);
            initialize();
        }
        vector<double> forward(vector<double>);
        void backward(vector<double>);
        void update();
        int get_seed();
        void set_seed(int);
    private:
        bool user_set_seed = false;
        int seed;
        void setGlobals(int, double);
        vector<double> result_buf;
        void stageNetwork(vector<double>);
        void createResultBuffer();
        void stageResults();
        vector<Layer> layers;
        void initialize();
        void final_pass();
};

extern double learningRate;
extern double beta1;
extern double beta2;
extern int epoch;
extern int optimizer;
extern int seed;

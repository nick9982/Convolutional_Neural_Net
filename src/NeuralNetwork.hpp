#include <stdarg.h>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#include <variant>
#include <string>
#include <cstdint>

#define noop (void)0

using namespace std;

class Weight
{
    private:
        int initialization, input, output;
        double value, alpha, m, v;
    public:
        Weight(int, int, int);
        void init();
        void update(double);
        void set_value(double);
        double get_value();
};

class Neuron
{
    private:
        double value, delta_value, cache_value;
        double (*act_function)(double);
        double (*act_function_derivative)(double);
    public:
        Neuron(int);
        void set_value_and_activate(double);
        void set_value(double);
        double get_value();
        void set_cache(double);
        double get_cache();
        void set_delta(double);
        double get_delta();
        double get_derivative();
};

class Bias
{
    private:
        double value, alpha, m, v;
        bool exists;
    public:
        Bias(bool);
        void update(double);
        void set_value(double);
        double get_value();
        void set_exists(bool);
        bool get_exists();
};

class StorageForNeuronsAndWeights
{
    public:
        vector<vector<Neuron*>> NeuronStore;
        vector<vector<Weight*>> WeightStore;
        vector<vector<Bias*>> BiasStore;
        void createDenseLayer(int, int, int, int, int);
};

vector<Neuron*> translateDoubleVecToNeuronVec(vector<double>);
int stringActivationToIntActivation(string);
int stringInitializationToIntInitialization(string);
int randomize();

class DenseLayer
{
    private:
        int in, out, activation, initialization, layerType, idx;
        bool hasBias;
    public:
        DenseLayer(int, string, string, bool=true);
        void init(int, int, int);
        void forward();
        void backward();
        void firstDeltas(vector<double>);
        void update();
        vector<Neuron> get_neurons();
        int getIn();
};

class ConvolutionalLayer
{
    private:
        vector<double> neurons;
        int x, y, input_channels, kernel_x, kernel_y, stride_x, stride_y, new_kernels, activation, initialization, layerType;
    public:
        ConvolutionalLayer(vector<int>, vector<int>, int, string, string);
        void init();
        vector<double> forward();
        vector<int> getIn();
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
                }
                else if(holds_alternative<ConvolutionalLayer*>(params[i]))
                {
                    Layer layer = get<ConvolutionalLayer*>(params[i]);
                    layers.push_back(layer);
                }
                else if(holds_alternative<PoolingLayer*>(params[i]))
                {
                    Layer layer = get<PoolingLayer*>(params[i]);
                    layers.push_back(layer);
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
};

extern StorageForNeuronsAndWeights globalStore;
extern double learningRate;
extern double beta1;
extern double beta2;
extern int epoch;
extern int optimizer;
extern int seed;

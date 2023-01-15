#include <stdarg.h>
#include <iostream>
#include <vector>
#include <variant>
#include <string>
#include <cstdint>

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
        void set_delta();
        double get_delta();
};

class StorageForNeuronsAndWeights
{
    public:
        vector<vector<Neuron*>> NeuronStore;
        vector<vector<Weight*>> WeightStore;
        void createDenseLayer(int, int, int, int);
};

vector<Neuron*> translateDoubleVecToNeuronVec(vector<double>);
int stringActivationToIntActivation(string);
int stringInitializationToIntInitialization(string);

class Bias
{
    private:
        double value, alpha, m, v;
        bool exists;
    public:
        void set_value(double);
        double get_value();
        void set_exists(bool);
        double get_exists();
};

class DenseLayer
{
    private:
        int in, out, activation, initialization, layerType, idx;
    public:
        DenseLayer(int, string, string);
        void init(int, int, int);
        void forward();
        void backward(vector<double> errors);
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

typedef variant<DenseLayer*, ConvolutionalLayer*, PoolingLayer*> Layer;

class NeuralNetwork
{
    public:
        template<typename... Args> NeuralNetwork(Args... args)
        {
            const int size = sizeof...(args);
            Layer params[size] = {args...};
            for(int i = 0; i < sizeof params / sizeof params[0]; i++)
            {
                layers.push_back(params[i]);
            }
            initialize();
        }
        vector<double> forward(vector<double>);
        void backward(vector<double>);
        void update();
    private:
        vector<double> result_buf;
        void stageNetwork(vector<double>);
        void createResultBuffer();
        void stageResults();
        vector<Layer> layers;
        void initialize();
};

extern StorageForNeuronsAndWeights globalStore;
extern int r;

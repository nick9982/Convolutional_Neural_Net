#include <stdarg.h>
#include <iostream>
#include <vector>
#include <variant>
#include <string>

using namespace std;

class Weight
{
    private:
        int initialization, input, output;
        double value, alpha, m, v;
    public:
        Weight(int, int, int);
        void update(double);
        void set_value(double);
        double get_value();
};

class Neuron
{
    private:
        double value, delta_value, cache_value;
        int activation;
        vector<Weight> weights;
    public:
        Neuron(int, int, int);
        void set_value(double);
        double get_value();
        void set_cache(double);
        double get_cache();
        void set_delta();
        double get_delta();
};

vector<Neuron> translateDoubleVecToNeuronVec(vector<double> vec);
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
        vector<Neuron> neurons;
        vector<Weight> weights;
        int in, out, activation, initialization, layerType;
    public:
        DenseLayer(int, string, string);
        void init(); //the reason init is separate from the constructor
        //is because some varaibles are assigned to this object after the
        //constructor is called. This way the user will not have to set
        //redundant parameters.
        vector<double> forward();
        void backward(vector<double> errors);
        void update();
};

class ConvolutionLayer
{
    private:
        vector<Neuron> neurons;
    public:
        ConvolutionLayer();
        void init();
        vector<double> forward();
        void backward();
        void update();
};

class PoolingLayer
{
    private:
    public:
        PoolingLayer();
};

class NeuralNetwork
{
    public:
        template<typename... Args> NeuralNetwork(Args... args)
        {
            const int size = sizeof...(args);
            /* DenseLayer* params[size] = {args...}; */
            /* for(int i = 0; i < sizeof params / sizeof params[0]; i++) */
            /* { */
            /*     layers.push_back(params[i]); */
            /* } */
            initialize();
        }
        vector<double> forward(vector<double>);
        void backward(vector<double>);
        void update();
    private:
        vector<DenseLayer*> layers;
        void initialize();
};

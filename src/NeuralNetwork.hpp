#include <stdarg.h>
#include <iostream>
#include <vector>
#include <variant>

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
        int in, out, activation, layerType;
    public:
        DenseLayer();
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
        vector<double> forward();
        void backward();
        void update();
};

class NeuralNetwork
{
    public:
        template<typename T, typename... Args> NeuralNetwork(const T& t, Args... args)
        {
            //Because constructor does not allow recursion I use a separate instanciate function
            //to unpack the packed parameters of the neural network.
            instanciate(t, args...);
        }
        vector<double> forward(vector<double>);
        void backward(vector<double>);
        void update();
    private:
        vector<variant<DenseLayer, ConvolutionLayer>> layers;
        template<typename T, typename... Args> void instanciate(const T& t, Args... args)
        {
            //This is a recursive function which will instanciate the neural network.
            cout << t << endl;
            if constexpr (sizeof...(Args) > 0)
            {
                instanciate(args...);
            }
        }
};

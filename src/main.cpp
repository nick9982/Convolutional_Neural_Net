#include "mnist_images.hpp"
#include "NeuralNetwork.hpp"
#include "DataMining/power_consumption.hpp"
#include <math.h>
#include <chrono>

void print_images(uchar**, int, int);
void learnPowerConsumption(bool, bool, int=0);
void test_network();
//OK we are going to recreate a faster dense network.
//We will elminate complex classes for neurons class
//We will use instead multiple arrays of data.
//The only classes we will keep is for layers and for
//the neural network itself. Global store of data will
//still be a thing. 
void test_net()
{
    NeuralNetwork nn(
        "Adam",
        0.1,
        new DenseLayer(3, "Linear", "0.5"),
        new DenseLayer(3, "ReLU", "0.5", false),
        new DenseLayer(3, "Linear", ""),
        118
    );

    vector<double> inp = {0.5, 0.5, 0.5};
    vector<double> out;
    out = nn.forward(inp);
    for(int i = 0; i < 9; i++)
    {
        cout << neuron_value[i] << endl;
    }
    
    vector<double> err = {5, -3, 2};
    nn.backward(err);
    cout << endl;
    for(int i = 3; i < 9; i++)
    {
        cout << delta_value[i] << endl;
    }
    
    nn.update();
    for(int i = 0; i < 18; i++)
    {
        cout << i << endl;
        cout << weight[i] << endl;
    }
    
    out = nn.forward({0.2, 0.2, 0.2});
    cout << endl;
    for(int i = 0; i < 9; i++)
    {
        cout << neuron_value[i] << endl;
    }
    
    nn.backward(err);
    cout << endl;
    for(int i = 3; i < 9; i++)
    {
        cout << delta_value[i] << endl;
    }
    
    nn.update();
    cout << endl;
    for(int i = 0; i < 18; i++)
    {
        cout << weight[i] << endl;
    }
}

int main (int argc, char *argv[])
{
    int number_of_images_test = 0, image_size_test = 0, number_of_images_train = 0, image_size_train = 0;
    int number_of_labels_test = 0, number_of_labels_train = 0;
    uchar** test_data = ImageDataset("../src/data/t10k-images-idx3-ubyte", number_of_images_test, image_size_test);
    uchar** train_data = ImageDataset("../src/data/train-images-idx3-ubyte", number_of_images_train, image_size_train);
    
    uchar* test_labels = labelDataset("../src/data/t10k-labels-idx1-ubyte", number_of_labels_test);
    uchar* train_labels = labelDataset("../src/data/train-labels-idx1-ubyte", number_of_labels_train);

    learnPowerConsumption(true, false);

    return 0;
}

void print_images(uchar** list_of_images, int number_of_images, int size_of_image)
{
    int xperimg = floor(sqrt(size_of_image));
    for(int i = 0; i < number_of_images; i++)
    {
        for(int j = 0; j < size_of_image; j++)
        {
            cout << +list_of_images[i][j] << " ";
            if(j % xperimg == 0) cout << endl;
        }
        cout << endl;
    }
}

void learnPowerConsumption(bool bias, bool isSeed, int seed)
{
    dataset processedData(32000, "../src/DataMining/data/tetuanCityPowerConsumption.csv", "Tetuan City Power Consumption");
    cout << "processing data" << endl;
    processedData.shuffle();

    vector<dataset> train_test_data = processedData.split(26000, "train_data", "test_data");
    dataset train_data = train_test_data[0];
    dataset test_data = train_test_data[1];
    if(!isSeed) seed = randomize();

    NeuralNetwork nn(
        "Adam",
        0.001,
        new DenseLayer(6, "Linear", "HeRandom"),
        new DenseLayer(10, "ReLU", "HeRandom", bias),
        new DenseLayer(5, "ReLU", "HeRandom", bias),
        new DenseLayer(3, "Linear", ""),
        seed
    );
    cout << nn.get_seed() << endl;
    vector<double> input(6, 0);
    vector<double> output(3, 0);

    double avg = 0;
    int avg_cnt = 0;
    double total = 0;
    int view_cnt = 1;
    cout << "testing initial performance..." << endl;



    for(uint i = 0; i < 200; i++)
    {
        for(uint j = 0; j < test_data.data[0].size(); j++)
        {
            if(j < input.size()) input[j] = test_data.data[i][j];
            else output[j-input.size()] = test_data.data[i][j];
        }
    
        vector<double> nno = nn.forward(input);
    
        double sum = 0;
        for(uint i = 0; i < output.size(); i++)
        {
            sum += abs(test_data.minMaxUnnormalization(nno[i], i+5) - test_data.minMaxUnnormalization(output[i], i+5));
        }
        avg_cnt++;
        total += sum/3;
        avg = total/avg_cnt;
    }

    cout << "training..." << endl;
    auto start = chrono::_V2::high_resolution_clock::now();
    for(uint i = 0; i < train_data.data.size(); i++)
    {
        for(uint j = 0; j < train_data.data[i].size(); j++)
        {
            if(j < input.size()) input[j] = train_data.data[i][j];
            else output[j-input.size()] = train_data.data[i][j];
        }
    
        nn.forward(input);
        nn.backward(output);
        nn.update();
    
        if(i % 1000 == 0) cout << "[" << i/1000 << "/"<<ceil(train_data.data.size()/1000)<<"]" << endl;
    }
    cout << "[" << ceil(train_data.data.size()/1000) << "/"<<ceil(train_data.data.size()/1000)<<"]" << endl;
    auto stop = chrono::_V2::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop-start);
    
    double error_before_training = avg;
    avg = 0;
    avg_cnt = 0;
    total = 0;
    view_cnt = 1;
    cout << "\ntesting final performance..." << endl;
    for(uint i = 0; i < test_data.data.size(); i++)
    {
        for(uint j = 0; j < test_data.data[0].size(); j++)
        {
            if(j < input.size()) input[j] = test_data.data[i][j];
            else output[j-input.size()] = test_data.data[i][j];
        }
    
        vector<double> nno = nn.forward(input);
    
        double sum = 0.f;
        for(uint i = 0; i < output.size(); i++)
        {
            sum += abs(test_data.minMaxUnnormalization(nno[i], i+5) - test_data.minMaxUnnormalization(output[i], i+5));
        }
        avg_cnt++;
        total += sum/3;
        avg = total/avg_cnt;
    }
    cout << "\nAverage error before training: " << error_before_training << endl;
    cout << "Average error after training: " << avg << endl;
    cout << "\nThe network's predictions are " << (1 - (avg/error_before_training)) * 100 << " percent more accurate than randomly choosing. " << endl;
    cout << "\nThe error is the average difference between the network's prediction of\nthe three region's power consumption and the actual power consumption." << endl;
    if(duration.count() * 0.000001 >= 60) cout << "\nTraining time: " << (duration.count() * 0.000001)/60 << " minutes" << endl;
    else cout << "\nTraining time: " << duration.count() * 0.000001 << " seconds" << endl;

}

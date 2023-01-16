#include "NeuralNetwork.hpp"
#include <chrono>

int randomize()
{
    auto start = chrono::_V2::high_resolution_clock::now();
    for(int i = 0; i < 300000; i++)
    {
        noop;
    }
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::microseconds>(end-start);
    return duration.count();
}

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <numeric>
#include <bit>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace std;
typedef unsigned char uchar;

int reverseInt(uint32_t& code);

uchar** ImageDataset(string full_path, int& number_of_images, int& image_size);

uchar* labelDataset(string full_path, int& number_of_labels);

class mnist_entropy_loss
{
    private:
        double *pred = new double[10];
        vector<vector<double>> dist = {{1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                       {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                                       {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                                       {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
                                       {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                                       {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
                                       {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                                       {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
                                       {0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
                                       {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}};
    public:
        mnist_entropy_loss();
        double* calculate(double*, char);
        double* calculateGradient(double*, char);
};

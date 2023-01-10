#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <numeric>
#include <bit>
#include <vector>
#include <algorithm>

using namespace std;
typedef unsigned char uchar;

int reverseInt(uint32_t& code);

uchar** ImageDataset(string full_path, int& number_of_images, int& image_size);

uchar* labelDataset(string full_path, int& number_of_labels);

#include "mnist_images.hpp"

int selected_idx;
int reverseInt(int code)
{
    unsigned char c1, c2, c3, c4;
    c1 = code & 255, c2 = (code >> 8) & 255, c3 = (code >> 16) & 255, c4 = (code >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

uchar** ImageDataset(string full_path, int& number_of_images, int& image_size)
{
    
    ifstream file(full_path, ios::binary);

    if(file.is_open())
    {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols  = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        uchar** dataset = 0;
        dataset = new uchar*[number_of_images];
        for(int i = 0; i < number_of_images; i++)
        {
            dataset[i] = new uchar[image_size];
            file.read((char *)dataset[i], image_size);
        }
        return dataset;
    }
    else
    {
        throw runtime_error("Cannot open file '" + full_path + "'!");
    }
}

uchar* labelDataset(string full_path, int& number_of_labels)
{
    ifstream file(full_path, ios::binary);

    int magic_number = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    if(magic_number != 2049) throw runtime_error("Invalid MNIST label file");

    file.read((char *)&number_of_labels, sizeof(int)), number_of_labels = reverseInt(number_of_labels);
    
    uchar* dataset = new uchar[number_of_labels];
    file.read((char *) dataset, number_of_labels);
    return dataset;
}

mnist_entropy_loss::mnist_entropy_loss(){}

double *mnist_entropy_loss::calculate(double *input, char label)
{
    vector<double> actual = this->dist[label];
    double* res = new double[10];
    for(int i = 0; i < 10; i++)
    {
        pred[i] = -(actual[i] * log(input[i]));
        res[i] = 0;
        if(actual[i] == 1) res[i] += 1; 
    }

    return res;
}

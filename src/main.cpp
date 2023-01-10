#include "mnist_images.hpp"
#include "NeuralNetwork.hpp"
#include <math.h>

void print_images(uchar**, int, int);

int main (int argc, char *argv[])
{
    int number_of_images_test = 0, image_size_test = 0, number_of_images_train = 0, image_size_train = 0;
    int number_of_labels_test = 0, number_of_labels_train = 0;
    uchar** test_data = ImageDataset("../src/data/t10k-images-idx3-ubyte", number_of_images_test, image_size_test);
    uchar** train_data = ImageDataset("../src/data/train-images-idx3-ubyte", number_of_images_train, image_size_train);
    
    uchar* test_labels = labelDataset("../src/data/t10k-labels-idx1-ubyte", number_of_labels_test);
    uchar* train_labels = labelDataset("../src/data/train-labels-idx1-ubyte", number_of_labels_train);

    /* print_images(test_data, number_of_images_test, image_size_test); */


    NeuralNetwork nn(18, 'a', 2, 3, 4);

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

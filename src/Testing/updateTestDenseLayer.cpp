#include "tests.hpp"
#include "../NeuralNetwork.hpp"

//The idea of this test is to see that the weights associated with the first DenseLayer
//get properly updated.

NeuralNetwork tstnn(
    "None",
    new DenseLayer(2, "Linear", "HeRandom"),
    new DenseLayer(3, "ReLU", "HeRandom")
);

void SettingUpUpdateDenseLayerTest()
{
    globalStore.NeuronStore[0][0]->set_value(2);
    globalStore.NeuronStore[0][1]->set_value(1);

    for(int i = 0 ; i < globalStore.WeightStore[0].size(); i++)
    {
        globalStore.WeightStore[0][i]->set_value(1.5);
    }

    globalStore.NeuronStore[1][0]->set_delta(0.2);
    globalStore.NeuronStore[1][1]->set_delta(0.1);
    globalStore.NeuronStore[1][2]->set_delta(0.2);
}

void updateTestDenseLayer()
{
    SettingUpUpdateDenseLayerTest();
    
    tstnn.update();
    
    for(int i = 0; i < globalStore.NeuronStore[0].size(); i++)
    {
        cout << globalStore.WeightStore[0][i]->get_value() << endl;
    }
}

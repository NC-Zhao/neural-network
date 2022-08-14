Course: CS-131-Artificial Intelligence
HW6: ANN
Name: Neal Zhao

There are three layers: input layer, hidden layer and output layer. I set the size of input layer as 4 since we have 4 features, . The size of hidden layer is 10. The size of output layer must be equal to the number of labels, so it is 3.

The bias is set to $1$, and the learning rate is $0.1$. 

The weights are randomly generated, and the train-validate-test set split is also random, the performance of the result is random as well. The accuracy score on the test set would vary each time the model is trained and tested. 

Each neuron uses a sigmoid activation function. 

The data is pre-processed using the following method:

- de-correlate use the variance method
- scale uniformly to range [0,1]

The pre-processing is done by `sklearn.preprocessing.StandardScaler` and `sklearn.preprocessing.MinMaxScaler`. No other sklearn methods are used in this program. 

This program also depends on the following packages:

- math, used for sigmoid computation. 
- numpy, used as basic data structure. 
- pandas, used for data reading. 


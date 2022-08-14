# CS 131 
# HW6
# Neal Zhao

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None

# the class that perform the ANN
class ANN:
    def __init__(self, learning_rate = 0.1, bias = 1, hidden_layer_size = 10):
        self.layer_size = [4, hidden_layer_size, 3]
        self.learning_rate = learning_rate
        self.bias = bias
        # bias weights, random initialization
        self.bias_w = [np.random.uniform(-1,1,4), np.random.uniform(-1,1, hidden_layer_size),
                     np.random.uniform(-1,1,3)]
        self.potentials = [np.zeros(4), np.zeros(hidden_layer_size), np.zeros(3)]
        self.outputs = [np.zeros(4), np.zeros(hidden_layer_size), np.zeros(3)]
        self.deltas = [np.zeros(4), np.zeros(hidden_layer_size), np.zeros(3)]
        # weights between layer
            # weights[0] has 4 arrays, 5 numbers each
            # weights[1] has 5 arrays, 3 numbers each
        self.weights = list()
        for parents_layer_i in range(2):
            layer_weights = []
            for parent_neuron_i in range(0, self.layer_size[parents_layer_i]):
                child_neuron_weights = np.random.uniform(-1, 1, 
                                                   self.layer_size[parents_layer_i + 1] )
                layer_weights.append(child_neuron_weights)
            self.weights.append(layer_weights)
        return
        
    
    
    # read data, encode the labels
    ## data should be named as 'data.txt' placed under the same file
    def get_data(self):
        df = pd.read_csv('data.txt', header = None)
        # encode the labels into 0 and 1
        def is_setosa(name):
            if name == 'Iris-setosa':
                return 1
            else: 
                return 0
        def is_versicolor(name):
            if name == 'Iris-versicolor':
                return 1
            else: 
                return 0
        def is_virginica(name):
            if name == 'Iris-virginica':
                return 1
            else: 
                return 0
        df['is_setosa'] = df[4].apply(lambda x: is_setosa(x))
        df['is_versicolor'] = df[4].apply(lambda x: is_versicolor(x))
        df['is_virginica'] = df[4].apply(lambda x: is_virginica(x))
        
        self.data = df
        self.data_size = len(df)
        return
    
    # split into three sets
    def split(self):
        split_index = np.arange(150)
        np.random.shuffle(split_index)
        self.train = self.data.iloc[split_index[:100]]
        self.test = self.data.iloc[split_index[100:125]]
        self.validation = self.data.iloc[split_index[125:150]]
    
    def pre_process(self):
        self.get_data()
        print('Read data success-----------------')
        self.split()
        self.StandardScaler = StandardScaler() # de-correlate
        self.StandardScaler.fit(self.train[[0,1,2,3]])
        self.train.at[:,[0,1,2,3]] = self.StandardScaler.transform(self.train[[0,1,2,3]])
        self.validation.at[:,[0,1,2,3]] = self.StandardScaler.transform(self.validation[[0,1,2,3]])
        
        self.MinMaxScaler = MinMaxScaler()
        self.MinMaxScaler.fit(self.train[[0,1,2,3]])
        self.train.at[:,[0,1,2,3]] = self.MinMaxScaler.transform(self.train[[0,1,2,3]])
        self.validation.at[:,[0,1,2,3]] = self.MinMaxScaler.transform(self.validation[[0,1,2,3]])
    
    def sigmoid(self, potential):
        
        # print('potential: ', potential)
        
        return 1 / (1 + math.exp(-potential))
    
    def forward(self, row):
        self.outputs[0] = np.array(row[0:4])
        # potentials
        for layer in range(1, 3):
            for neuron in range(0, self.layer_size[layer]):
                self.potentials[layer] = self.bias_w[layer] * self.bias
                for parent in range(0, self.layer_size[layer - 1]):
                    self.potentials[layer][neuron] += (self.weights[layer-1][parent][neuron] * self.outputs[layer - 1][parent])
                self.outputs[layer][neuron] = self.sigmoid(self.potentials[layer][neuron])
                
    
    def backward(self, row):
        labels = row[[5,6,7]]
            
        self.deltas[2] = self.outputs[2] * (1 - self.outputs[2]) * (labels - self.outputs[2])
        # compute the error
        for layer in range(1, 0, -1):
            for parent in range(0, self.layer_size[layer]):
                self.deltas[layer][parent] = 0
                self.deltas[layer][parent] += (self.weights[layer][parent] * self.deltas[layer + 1]).sum()
                self.deltas[layer][parent] *= self.outputs[layer][parent] * (1 - self.outputs[layer][parent])
        
        for layer in range(2, 0, -1):
            for child in range(0, self.layer_size[layer]):
                child_delta = self.deltas[layer][child]
                self.bias_w[layer][child] += self.learning_rate * self.bias * child_delta
                for parent in range(0, self.layer_size[layer - 1]):
                    self.weights[layer - 1][parent][child] += self.learning_rate * self.outputs[layer - 1][parent] * child_delta
    
    def result(self):
        setosa = self.outputs[2][0]
        versicolour = self.outputs[2][1]
        virginica = self.outputs[2][2]
        max_output = max(setosa, versicolour, virginica)
        if setosa == max_output:
            return 0
        elif versicolour == max_output:
            return 1
        elif virginica == max_output:
            return 2
    
    
    # compute the average mse
    def validation_err(self):
        error = 0
        for i in range(len(self.validation)):
            row_v = self.validation.iloc[i].values
            self.forward(row_v)
            error += (row_v[5] - self.outputs[2][0])**2
            error += (row_v[6] - self.outputs[2][1])**2
            error += (row_v[7] - self.outputs[2][2])**2
        return error / len(self.validation)
    
    def train(self):
        self.pre_process()
        previous_error = 0
        print('training_start-----------------')
        
        for itr in range(0, 500):
            for i in range(len(self.train)):
                row = self.train.iloc[i].values
                self.forward(row)
                self.backward(row)
            current_error = self.validation_err()
            if abs(current_error - previous_error) / current_error < 0.0001:
                break
            previous_error = current_error
        print("Training ends-----------------")
    
    def classify(self, row):
        row1 = self.StandardScaler.transform(row)
        row2 = self.MinMaxScaler.transform(row1)
        self.forward(row2[0])
        result = self.result()
        return result
    
    def test_set(self):
        correct = 0
        for i in range(len(self.test)):
            predicted = self.classify(self.test.iloc[i][0:4].values.reshape(1,-1))
            if self.test.iloc[i, predicted + 5] == 1:
                correct += 1
        print("The test accuracy is {:6.2f} percent".format(correct/25*100))
    
if __name__ == "__main__":
    print("Max iteration in training step is 1000 and acceptable mse is 0.0001, training ends if one of the condition is met")
    print("The weights are initialized in random fashion, you may get different accuracy for different trials")
    print("This ANN use learning rate = 0.1, bias = 1")
    print()
    ann = ANN()
    ann.train()
    ann.test_set()
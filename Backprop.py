# Author : Alok Dixit
# Date : 24-may-2018

"""
Objective: To implement a Backprop demo
            1. We should be able to specify the number of layers in Neural net
            2. A list of number of neurons in each layer
            3. We should be able to specify the activation function
"""

import numpy as np
import pandas as pd

def elem_wise_multiply_and_add(input, weights, bias):
    return (sum([i*w for i, w in zip(input, weights)]) + bias)


def apply_activation_function(output, func):
    return [func(output_i) for output_i in output]

def RMSE(output, labels):
    return sum([((out-label)**2)/2 for out, label in zip(output, labels)])**0.5

class Neuron(object):
    
    def __init__(self, w, b, neuron_id):
        self.w = w
        self.b = b
        self.next_layer_grad = 0
        self.net = None
        self.out = None
        self.neuron_id = neuron_id

    def update_weights(self, grad):
        self.next_layer_grad = grad
        self.w = self.w - alpha*self.next_layer_grad

INPUT_DIM = 2
num_of_hidden_layers = 2
OUTPUT_DIM = 1

NUM_OF_ITERATION = 5

num_of_neurons_hidden_layer = [3, 3]

# Learning Rate
alpha = 0.1

assert (len(num_of_neurons_hidden_layer) == num_of_hidden_layers), "Please specify the number of neurons for each layer. expected {0}, got {1}".format(num_of_hidden_layers, len(num_of_neurons_hidden_layer))

activation_function = None
activation_grad = None

def RELU(input):
    # RELU
    return (input if input > 0 else 0)

def RELU_grad(input):
    return 1 if input > 0 else 0

activation_function = RELU
activation_grad = RELU_grad

layers = []
layers.append(list(Neuron(1, 0.1, "input_{}".format(i)) for i in range((INPUT_DIM))))

for idx, num_of_hidden_neuron in enumerate(num_of_neurons_hidden_layer):
    last_layer_neurons = INPUT_DIM if idx == 0 else num_of_neurons_hidden_layer[idx-1]
    layers.append([Neuron(np.random.uniform(-2,2,(last_layer_neurons,1)), 0.1,"hidden{}_{}".format(idx, i)) for i in range(num_of_hidden_neuron)])
    # layers.append([{"w":np.random.uniform(-2,2,(last_layer_neurons,1)), "b":0.1,"grad":0, "neuron_id":"hidden{}_{}".format(idx, i)} for i in range(num_of_hidden_neuron)])



layers.append(list(Neuron(np.random.uniform(-2,2,(num_of_neurons_hidden_layer[-1],1)), 0.1, "output_{}".format(i)) for i in range(OUTPUT_DIM)))
# layers.append(list({"w":np.random.uniform(-2,2,(num_of_neurons_hidden_layer[-1],1)), "b":0.1,"grad":0, "neuron_id":"output_{}".format(i)} for i in range(OUTPUT_DIM)))


df = pd.read_csv("DATA.csv")

input_x = df["x"]
input_y = df["y"]

label = df["z"]

for each_iteration in range(NUM_OF_ITERATION):
    for x, y in zip(input_x,input_y):
        h1_input = (x*layers[0][0].w, y*layers[0][1].w)
        
        layer_i_input = h1_input
        for layer_i in layers[1:-1]:
            hidden_out = []
            for neuron_i in layer_i:
                hidden_out.append(elem_wise_multiply_and_add(layer_i_input, neuron_i.w, neuron_i.b))
            hidden_out = apply_activation_function(hidden_out, activation_function)
            layer_i_input = hidden_out

        output_layer_input = layer_i_input
        output = []
        for neuron_i in layers[-1]:
            output.append(elem_wise_multiply_and_add(output_layer_input, neuron_i.w, neuron_i.b))
            output = apply_activation_function(output, activation_function)
        # Now we will calculate the error and update the weights

        loss = RMSE(output, label)
        # print("iteration = {0}, loss = {1}".format(each_iteration, loss))
        


        



        









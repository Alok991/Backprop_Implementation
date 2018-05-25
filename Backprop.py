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

INPUT_DIM = 2
num_of_hidden_layers = 2
OUTPUT_DIM = 1

num_of_neurons_hidden_layer = [3, 3]

assert (len(num_of_neurons_hidden_layer) == num_of_hidden_layers), "Please specify the number of neurons for each layer. expected {0}, got {1}".format(num_of_hidden_layers, len(num_of_neurons_hidden_layer))

activation_function = None

def RELU(input):
    # RELU
    return (input if input > 0 else 0)

activation_function = RELU

layers = []
layers.append(list({"w":1, "b":0.1, "neuron_id":"input_{}".format(i)} for i in range((INPUT_DIM))))

for idx, num_of_hidden_neuron in enumerate(num_of_neurons_hidden_layer):
    last_layer_neurons = INPUT_DIM if idx == 0 else num_of_neurons_hidden_layer[idx-1]
    layers.append([{"w":np.random.rand(last_layer_neurons), "b":0.1, "neuron_id":"hidden{}_{}".format(idx, i)} for i in range(num_of_hidden_neuron)])



layers.append(list({"w":np.random.rand(num_of_neurons_hidden_layer[-1]), "b":0.1, "neuron_id":"output_{}".format(i)} for i in range(OUTPUT_DIM)))


df = pd.read_csv("DATA.csv")

input_x = df["x"]
input_y = df["y"]

label = df["z"]

for x, y in zip(input_x,input_y):
    h1_input = (x*layers[0][0]["w"], y*layers[0][1]["w"])
    # print("output of layer_0 is {} and will act as inputs to hidden layer 1".format(str(h1_input)))
    
    layer_i_input = h1_input
    for layer_i in layers[1:-1]:
        hidden_out = []
        for neuron_i in layer_i:
            hidden_out.append(elem_wise_multiply_and_add(layer_i_input, neuron_i["w"], neuron_i["b"]))
        hidden_out = apply_activation_function(hidden_out, activation_function)
        layer_i_input = hidden_out

    output_layer_input = layer_i_input
    output = []
    for neuron_i in layers[-1]:
        output.append(elem_wise_multiply_and_add(output_layer_input, neuron_i["w"], neuron_i["b"]))

    print("final output for input = ({0}, {1}) is {2}".format(x, y, str(output)))

    # Now we will calculate the error and update the weights

    err = RMSE(output, label)
    print(err)







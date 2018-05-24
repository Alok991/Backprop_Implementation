# Author : Alok Dixit
# Date : 24-may-2018

"""
Objective: To implement a Backprop demo
            1. We should be able to specify the number of layers in Neural net
            2. A list of number of neurons in each layer
            3. We should be able to specify the activation function
"""

import numpy as np


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
layers.append(list([{"w":1, "b":0, "neuron_id":"input_{}".format(i)}] for i in range((INPUT_DIM))))

for idx, num_of_hidden_neuron in enumerate(num_of_neurons_hidden_layer):
    last_layer_neurons = INPUT_DIM if idx == 0 else num_of_neurons_hidden_layer[idx-1]
    layers.append([{"w":np.random.rand(last_layer_neurons), "b":0, "neuron_id":"hidden{}_{}".format(idx, i)} for i in range(num_of_hidden_neuron)])



layers.append(list({"w":np.random.rand(num_of_neurons_hidden_layer[-1]), "b":0, "neuron_id":"output_{}".format(i)} for i in range(OUTPUT_DIM)))



for layer in layers:
    print(layer)
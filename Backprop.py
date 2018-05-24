# Author : Alok Dixit
# Date : 24-may-2018

"""
Objective: To implement a Backprop demo
            1. We should be able to specify the number of layers in Neural net
            2. A list of number of neurons in each layer
            3. We should be able to specify the activation function
"""

LAYER = 5
num_of_neurons = [2, 3, 4, 3, 1]

assert (len(num_of_neurons) == LAYER), "Please specify the number of neurons for each layer. expected {0}, got {1}".format(LAYER, len(num_of_neurons))

activation_function = None

def RELU(input):
    # RELU
    return (input if input > 0 else 0)

activation_function = RELU




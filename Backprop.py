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

# +1 for bias node
INPUT_LAYER_DIM = 2 + 1
HIDDEN_LAYER_DIM = 3
OUTPUT_LAYER_DIM = 1

NUM_OF_HIDDEN_LAYERS = 2

NUM_OF_ITERATION = 5


# Learning Rate
alpha = 0.1

activation_function = None
activation_grad = None

def RELU(input):
    # RELU
    input = input.copy()
    input[input<0]=0
    return input

def RELU_grad(input):
    # RELU_grad
    input = input.copy()
    input[input<=0]=0
    input[input>0]=1
    return input

activation_function = RELU
activation_grad = RELU_grad

df = pd.read_csv("DATA.csv")

input_xy = [(x,y, 1) for x, y in zip(df["x"], df["y"])]
label = df["z"]

# This will store the output of activation function of neurons
input_out = np.zeros(shape=(INPUT_LAYER_DIM))+1
hidden_out = np.zeros(shape=(NUM_OF_HIDDEN_LAYERS, HIDDEN_LAYER_DIM))+1
output_out = np.zeros(shape=(OUTPUT_LAYER_DIM))+1


def make_random_array(a, *b):
    return np.random.randint(-100,100, size=(a,*b)).astype(np.float)/100


# These are the weights between layers

# w[0] = first neuron of input layer to all neurons on hidden layer
w_i_h1 = make_random_array(INPUT_LAYER_DIM, HIDDEN_LAYER_DIM)

# w[0] = 2D matrix of weights between hi and hj hidden layers
w_hi_hj = make_random_array(NUM_OF_HIDDEN_LAYERS, HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM)

# w[0] = first neuron of hidden layer to all neurons on output layer
w_h_o = make_random_array(HIDDEN_LAYER_DIM, OUTPUT_LAYER_DIM)

# c[0] = first neuron of input layer to all neurons on hidden layer
c_i_h1 = make_random_array(INPUT_LAYER_DIM, HIDDEN_LAYER_DIM)

# c[0] = 2D matrix of weights between hi and hj hidden layers
c_hi_hj = make_random_array(NUM_OF_HIDDEN_LAYERS, HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM)

# c[0] = first neuron of hidden layer to all neurons on output layer
c_h_o = make_random_array(HIDDEN_LAYER_DIM, OUTPUT_LAYER_DIM)


for i in range(NUM_OF_ITERATION):
    for j, inp in enumerate(input_xy):
        # output of input layer(same as input to network)
        input_out = np.array(inp)

        # output of hidden layers
        for k, hidden_layer in enumerate(hidden_out):
            if k == 0:
                # out hidden_layer 1
                hidden_out[k] = activation_function((input_out.dot(w_i_h1)))
            else:
                # out layers ...
                hidden_out[k] = activation_function((hidden_out[k-1].dot(w_hi_hj[k])))

        # output of last layer
        output_out = (hidden_out[-1].dot(w_h_o))

        """
        BACKPROPOGATION START HERE !!
        """


        
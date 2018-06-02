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
import math

# +1 for bias node
INPUT_LAYER_DIM = 2 + 1
HIDDEN_LAYER_DIM = 3
OUTPUT_LAYER_DIM = 1

NUM_OF_HIDDEN_LAYERS = 2

NUM_OF_ITERATION = 10000


# Learning Rate
alpha = 0.01

momentum = 0.01

activation_function = None
activation_grad = None


def nearest_int(x):
    return 1 if x>0.5 else 0

def tanh(input):
    return np.array([math.tanh(inp) for inp in input])

def tanh_grad(input):
    return np.array([(1.0 - inp**2) for inp in input])

def RELU(input):
    return [x if x>0 else 0 for x in input]

def RELU_grad(input):
    return [1 if x>0 else 0 for x in input]

activation_function = tanh
activation_grad = tanh_grad

df = pd.read_csv("DATA.csv")

input_xy = [(x,y, 1) for x, y in zip(df["x"], df["y"])]
label_z = df["z"]

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
w_hi_hj = make_random_array(NUM_OF_HIDDEN_LAYERS-1, HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM)

# w[0] = first neuron of hidden layer to all neurons on output layer
w_h_o = make_random_array(HIDDEN_LAYER_DIM, OUTPUT_LAYER_DIM)

# c[0] = first neuron of input layer to all neurons on hidden layer
c_i_h1 = make_random_array(INPUT_LAYER_DIM, HIDDEN_LAYER_DIM)

# c[0] = 2D matrix of weights between hi and hj hidden layers
c_hi_hj = make_random_array(NUM_OF_HIDDEN_LAYERS-1, HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM)

# c[0] = first neuron of hidden layer to all neurons on output layer
c_h_o = make_random_array(HIDDEN_LAYER_DIM, OUTPUT_LAYER_DIM)


print("Learning...")
for i in range(NUM_OF_ITERATION):
    alpha = alpha/(i+1)
    for j, inp in enumerate(input_xy):
        # output of input layer(same as input to network)
        label = label_z[j]
        input_out = np.array(inp)

        # output of hidden layers
        for k, hidden_layer in enumerate(hidden_out):
            if k == 0:
                # out hidden_layer 1
                hidden_out[k] = activation_function((input_out.dot(w_i_h1)))
            else:
                hidden_out[k] = activation_function((hidden_out[k-1].dot(w_hi_hj[k-1])))

        # output of last layer
        output_out = activation_function(hidden_out[-1].dot(w_h_o))

        """
        BACKPROPOGATION START HERE !!
        """
        hidden_delta = []
        # error delta for output layer
        err = output_out - label
        output_delta = activation_grad(output_out)*err

        # error delta for hidden layers
        next_layer_delta = output_delta
        for k, hidden_layer in enumerate(reversed(hidden_out)):
            k = len(hidden_out) - k - 1
            # last hidden layer
            if k == len(hidden_out)-1:
                err = next_layer_delta.dot(w_h_o.T)
            else:
                err = next_layer_delta.dot(w_hi_hj[k].T)
            
            this_hidden_layer_delta = activation_grad(hidden_out[k])*err
            next_layer_delta = this_hidden_layer_delta
            hidden_delta.insert(0, this_hidden_layer_delta)

        hidden_delta = np.array(hidden_delta)
        
        # update the intput weights ==> w_i_h1
        w_i_h1 = w_i_h1 - momentum*c_i_h1 - alpha*(input_out.reshape((-1,1)).dot(hidden_delta[0].reshape((1,-1))))
        c_i_h1 = input_out.reshape((-1,1)).dot(hidden_delta[0].reshape((1,-1)))

        # update inter-hidden layer weights
        for k in range(len(w_hi_hj)):
            w_hi_hj[k] = w_hi_hj[k] - momentum*c_hi_hj[k] - alpha*((hidden_out[k].T).dot(hidden_delta[k]))
            c_hi_hj[k] = (hidden_out[k].T).dot(hidden_delta[k])
    if i%1000 == 0 :
        print("iteration {0} completed".format(i))




print("After Learning...")
for j, inp in enumerate(input_xy):
    # output of input layer(same as input to network)
    label = label_z[j]
    input_out = np.array(inp)

    # output of hidden layers
    for k, hidden_layer in enumerate(hidden_out):
        if k == 0:
            # out hidden_layer 1
            hidden_out[k] = activation_function((input_out.dot(w_i_h1)))
        else:
            hidden_out[k] = activation_function((hidden_out[k-1].dot(w_hi_hj[k-1])))

    # output of last layer
    output_out = (hidden_out[-1].dot(w_h_o))
    err = output_out - label
    print("Expected output => {0}, Real output => {1}".format(label, nearest_int(output_out)))

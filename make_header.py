"""
Script for making a C++ headerfile with LSTM network
based on a keras ".h5" model placed in MODEL_PATH
(see config file).
"""

import numpy as np
import pandas as pd
import keras

from keras.models import load_model

from config import *
from preprocessing import get_convoluted_data

LAYER_DENSE = 1
LAYER_ACTIVATION = 2
LAYER_LSTM = 3

ACTIVATION_LINEAR = 1
ACTIVATION_RELU = 2
ACTIVATION_SOFTPLUS = 3
ACTIVATION_SIGMOID = 4
ACTIVATION_TANH = 5
ACTIVATION_HARD_SIGMOID = 6

def activation_type(activation):
    if activation == 'linear':
        return ACTIVATION_LINEAR
    elif activation == 'relu':
        return ACTIVATION_RELU
    elif activation == 'softplus':
        return ACTIVATION_SOFTPLUS
    elif activation == 'tanh':
        return ACTIVATION_TANH
    elif activation == 'sigmoid':
        return ACTIVATION_SIGMOID
    elif activation == 'hard_sigmoid':
        return ACTIVATION_HARD_SIGMOID

def append_number_vector(file, name, vector):
    file.write("std::vector<float> " + name + " = {\n")
    file.write(', '.join(map(str, vector.tolist())))
    file.write("};\n\n")

def append_includes(file):
    file.write("#include <string>\n")
    file.write("#include <vector>\n")
    file.write("\n")

def make_header(model):
    model_layers = [l for l in model.layers if type(l).__name__ not in ['Dropout']]
    num_layers = len(model_layers)
    with open(HEADERFILE_NAME, "a") as file:
        append_includes(file)
        file.write("unsigned int NUM_LAYERS = " + str(num_layers) + ";\n")
        if(num_layers != 2):
            print("Models with two layers are allowed only")
            raise ValueError

        for layer in model_layers:
            layer_type = type(layer).__name__

            if layer_type == 'Dense':
                weights = layer.get_weights()[0]
                biases = layer.get_weights()[1]
                activation = layer.get_config()['activation']

                file.write("unsigned int DENSE_WEIGHTS_ROWS = " + str(weights.shape[0]) + ";\n")
                file.write("unsigned int DENSE_WEIGHTS_COLS = " + str(weights.shape[1]) + ";\n")
                file.write("unsigned int DENSE_BIASES_SHAPE = " + str(biases.shape[0]) + ";\n")
                file.write("unsigned int DENSE_ACTIVATION = " + str(activation_type(activation)) + ";\n\n")

                append_number_vector(file, "DENSE_WEIGHTS", weights.flatten())
                append_number_vector(file, "DENSE_BIASES", biases)


            elif layer_type == 'LSTM':
                recurrent_activation = layer.get_config()['recurrent_activation']
                activation = layer.get_config()['activation']
                return_sequences = int(layer.get_config()['return_sequences'])

                file.write("unsigned int LSTM_RECURRENT_ACTIVATION = " + str(activation_type(recurrent_activation)) + ";\n")
                file.write("unsigned int LSTM_ACTIVATION =  " + str(activation_type(activation)) + ";\n")
                file.write("unsigned int RETURN_SEQUENCES = " + str(return_sequences) + ";\n")

                W = model.layers[0].get_weights()[0]
                U = model.layers[0].get_weights()[1]
                b = model.layers[0].get_weights()[2]
                num_units = int(int(model.layers[0].trainable_weights[0].shape[1])/4)
                file.write("unsigned int N_HIDDEN_NEURONS = " + str(num_units) + ";\n\n")

                W_i = W[:, :num_units]
                W_f = W[:, num_units: num_units * 2]
                W_c = W[:, num_units * 2: num_units * 3]
                W_o = W[:, num_units * 3:]

                U_i = U[:, :num_units]
                U_f = U[:, num_units: num_units * 2]
                U_c = U[:, num_units * 2: num_units * 3]
                U_o = U[:, num_units * 3:]

                b_i = b[:num_units]
                b_f = b[num_units: num_units * 2]
                b_c = b[num_units * 2: num_units * 3]
                b_o = b[num_units * 3:]

                file.write("unsigned int W_i_ROWS = " + str(W_i.shape[0]) + ";\n")
                file.write("unsigned int W_i_COLS = " + str(W_i.shape[1]) + ";\n")
                file.write("unsigned int W_f_ROWS = " + str(W_f.shape[0]) + ";\n")
                file.write("unsigned int W_f_COLS = " + str(W_f.shape[1]) + ";\n")
                file.write("unsigned int W_c_ROWS = " + str(W_c.shape[0]) + ";\n")
                file.write("unsigned int W_c_COLS = " + str(W_c.shape[1]) + ";\n")
                file.write("unsigned int W_o_ROWS = " + str(W_o.shape[0]) + ";\n")
                file.write("unsigned int W_o_COLS = " + str(W_o.shape[1]) + ";\n")
                file.write("unsigned int U_i_ROWS = " + str(U_i.shape[0]) + ";\n")
                file.write("unsigned int U_i_COLS = " + str(U_i.shape[1]) + ";\n")
                file.write("unsigned int U_f_ROWS = " + str(U_f.shape[0]) + ";\n")
                file.write("unsigned int U_f_COLS = " + str(U_f.shape[1]) + ";\n")
                file.write("unsigned int U_c_ROWS = " + str(U_c.shape[0]) + ";\n")
                file.write("unsigned int U_c_COLS = " + str(U_c.shape[1]) + ";\n")
                file.write("unsigned int U_o_ROWS = " + str(U_o.shape[0]) + ";\n")
                file.write("unsigned int U_o_COLS = " + str(U_o.shape[1]) + ";\n")
                file.write("unsigned int b_i_SHAPE = " + str(b_i.shape[0]) + ";\n")
                file.write("unsigned int b_f_SHAPE = " + str(b_f.shape[0]) + ";\n")
                file.write("unsigned int b_c_SHAPE = " + str(b_c.shape[0]) + ";\n")
                file.write("unsigned int b_o_SHAPE = " + str(b_o.shape[0]) + ";\n")
                file.write("\n")

                append_number_vector(file, "W_i", W_i.flatten())
                append_number_vector(file, "W_f", W_f.flatten())
                append_number_vector(file, "W_c", W_c.flatten())
                append_number_vector(file, "W_o", W_o.flatten())
                append_number_vector(file, "U_i", U_i.flatten())
                append_number_vector(file, "U_f", U_f.flatten())
                append_number_vector(file, "U_c", U_c.flatten())
                append_number_vector(file, "U_o", U_o.flatten())

                append_number_vector(file, "b_i", b_i)
                append_number_vector(file, "b_f", b_f)
                append_number_vector(file, "b_c", b_c)
                append_number_vector(file, "b_o", b_o)
    file.close()

if __name__ == '__main__':
    model = load_model(MODEL_PATH)
    make_header(model)

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

"""
Given activation return a number assigned to that activation.
"""
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

"""
Append functions are used to append strings to the "file",
so that it has a C/C++ headerfile style.
"""
def append_vector(file, type, name, vector_object):
    file.write("std::vector<{}> {} = {{\n".format(type, name))
    file.write(', '.join(map(str, vector_object.tolist())))
    file.write("};\n\n")

def append_list(file, type, name, list):
    file.write("std::vector<{}> {} = {{".format(type, name))
    file.write(', '.join(map(str, list)))
    file.write("};\n")

def append_vector_of_vectors(file, n, type, name, array):
    file.write("std::vector< std::vector<{}> > {} = {{\n".format(type, name))
    for i in range(n):
        append_numbers(file, array[i].flatten())
        if(i != n - 1):
            file.write(",\n")
    file.write("\n};\n")

def append_numbers(file, array):
    file.write("{")
    file.write(', '.join(map(str, array.tolist())))
    file.write("}")

def append_includes(file):
    file.write("#include <string>\n")
    file.write("#include <vector>\n")
    file.write("\n")

"""
Function used for creating and saving to disk a headerfile.
Input: Keras model (.h5)
Output: None (saves file to disk)
"""
def make_header(model):
    model_layers = [l for l in model.layers if type(l).__name__ not in ['Dropout']]
    num_layers = len(model_layers)

    # Count number of each layer type
    lstm_layers_count = 0
    dense_layers_count = 0

    with open(HEADERFILE_NAME, "a") as file:
        append_includes(file)
        file.write("unsigned int NUM_LAYERS = " + str(num_layers) + ";\n")

        if len([type(layer).__name__ for layer in model_layers if type(layer).__name__ == 'Dense']) > 1:
            print("Error: Only one dense layer allowed")
            exit()

        if 'LSTM' in [type(layer).__name__ for layer in model_layers]:
            recurrent_activation_list = []
            activation_list = []
            return_sequences_list = []

            num_units_list = []

            W_i_list = []
            W_f_list = []
            W_c_list = []
            W_o_list = []

            U_i_list = []
            U_f_list = []
            U_c_list = []
            U_o_list = []

            b_i_list = []
            b_f_list = []
            b_c_list = []
            b_o_list = []

            W_i_ROWS_list = []
            W_i_COLS_list = []
            W_f_ROWS_list = []
            W_f_COLS_list = []
            W_c_ROWS_list = []
            W_c_COLS_list = []
            W_o_ROWS_list = []
            W_o_COLS_list = []
            U_i_ROWS_list = []
            U_i_COLS_list = []
            U_f_ROWS_list = []
            U_f_COLS_list = []
            U_c_ROWS_list = []
            U_c_COLS_list = []
            U_o_ROWS_list = []
            U_o_COLS_list = []
            b_i_SHAPE_list = []
            b_f_SHAPE_list = []
            b_c_SHAPE_list = []
            b_o_SHAPE_list = []

        for layer in model_layers:
            layer_type = type(layer).__name__
            if layer_type == 'Dense':
                dense_layers_count += 1

                weights = layer.get_weights()[0]
                biases = layer.get_weights()[1]
                activation = layer.get_config()['activation']

                file.write("unsigned int DENSE_WEIGHTS_ROWS = " + str(weights.shape[0]) + ";\n")
                file.write("unsigned int DENSE_WEIGHTS_COLS = " + str(weights.shape[1]) + ";\n")
                file.write("unsigned int DENSE_BIASES_SHAPE = " + str(biases.shape[0]) + ";\n")
                file.write("unsigned int DENSE_ACTIVATION = " + str(activation_type(activation)) + ";\n\n")

                append_vector(file, "float", "DENSE_WEIGHTS", weights.flatten())
                append_vector(file, "float", "DENSE_BIASES", biases)

            elif layer_type == 'LSTM':
                lstm_layers_count += 1

                recurrent_activation = layer.get_config()['recurrent_activation']
                activation = layer.get_config()['activation']
                return_sequences = int(layer.get_config()['return_sequences'])

                recurrent_activation_list += str(activation_type(recurrent_activation))
                activation_list += str(activation_type(activation))
                return_sequences_list += str(return_sequences)

                W = layer.get_weights()[0]
                U = layer.get_weights()[1]
                b = layer.get_weights()[2]

                num_units = int(int(layer.trainable_weights[0].shape[1])/4)
                num_units_list.append(num_units)
                # file.write("unsigned int N_HIDDEN_NEURONS = " + str(num_units) + ";\n\n")

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

                W_i_list.append(W_i)
                W_f_list.append(W_f)
                W_c_list.append(W_c)
                W_o_list.append(W_o)

                U_i_list.append(U_i)
                U_f_list.append(U_f)
                U_c_list.append(U_c)
                U_o_list.append(U_o)

                b_i_list.append(b_i)
                b_f_list.append(b_f)
                b_c_list.append(b_c)
                b_o_list.append(b_o)

                W_i_ROWS_list.append(str(W_i.shape[0]))
                W_i_COLS_list.append(str(W_i.shape[1]))
                W_f_ROWS_list.append(str(W_f.shape[0]))
                W_f_COLS_list.append(str(W_f.shape[1]))
                W_c_ROWS_list.append(str(W_c.shape[0]))
                W_c_COLS_list.append(str(W_c.shape[1]))
                W_o_ROWS_list.append(str(W_o.shape[0]))
                W_o_COLS_list.append(str(W_o.shape[1]))
                U_i_ROWS_list.append(str(U_i.shape[0]))
                U_i_COLS_list.append(str(U_i.shape[1]))
                U_f_ROWS_list.append(str(U_f.shape[0]))
                U_f_COLS_list.append(str(U_f.shape[1]))
                U_c_ROWS_list.append(str(U_c.shape[0]))
                U_c_COLS_list.append(str(U_c.shape[1]))
                U_o_ROWS_list.append(str(U_o.shape[0]))
                U_o_COLS_list.append(str(U_o.shape[1]))
                b_i_SHAPE_list.append(str(b_i.shape[0]))
                b_f_SHAPE_list.append(str(b_f.shape[0]))
                b_c_SHAPE_list.append(str(b_c.shape[0]))
                b_o_SHAPE_list.append(str(b_o.shape[0]))

        append_list(file, "unsigned int", 'LSTM_RECURRENT_ACTIVATION', recurrent_activation_list)
        append_list(file, "unsigned int", 'LSTM_ACTIVATION', activation_list)
        append_list(file, "unsigned int", 'RETURN_SEQUENCES', return_sequences_list)
        append_list(file, "unsigned int", 'N_HIDDEN_NEURONS', num_units_list)
        append_list(file, "unsigned int", 'W_i_ROWS', W_i_ROWS_list)
        append_list(file, "unsigned int", 'W_i_COLS', W_i_COLS_list)
        append_list(file, "unsigned int", 'W_f_ROWS', W_f_ROWS_list)
        append_list(file, "unsigned int", 'W_f_COLS', W_f_COLS_list)
        append_list(file, "unsigned int", 'W_c_ROWS', W_c_ROWS_list)
        append_list(file, "unsigned int", 'W_c_COLS', W_c_COLS_list)
        append_list(file, "unsigned int", 'W_o_ROWS', W_o_ROWS_list)
        append_list(file, "unsigned int", 'W_o_COLS', W_o_COLS_list)
        append_list(file, "unsigned int", 'U_i_ROWS', U_i_ROWS_list)
        append_list(file, "unsigned int", 'U_i_COLS', U_i_COLS_list)
        append_list(file, "unsigned int", 'U_f_ROWS', U_f_ROWS_list)
        append_list(file, "unsigned int", 'U_f_COLS', U_f_COLS_list)
        append_list(file, "unsigned int", 'U_c_ROWS', U_c_ROWS_list)
        append_list(file, "unsigned int", 'U_c_COLS', U_c_COLS_list)
        append_list(file, "unsigned int", 'U_o_ROWS', U_o_ROWS_list)
        append_list(file, "unsigned int", 'U_o_COLS', U_o_COLS_list)
        append_list(file, "unsigned int", 'b_i_SHAPE', b_i_SHAPE_list)
        append_list(file, "unsigned int", 'b_f_SHAPE', b_f_SHAPE_list)
        append_list(file, "unsigned int", 'b_c_SHAPE', b_c_SHAPE_list)
        append_list(file, "unsigned int", 'b_o_SHAPE', b_o_SHAPE_list)

        append_vector_of_vectors(file, lstm_layers_count ,'float', 'W_i', W_i_list)
        append_vector_of_vectors(file, lstm_layers_count ,'float', 'W_f', W_f_list)
        append_vector_of_vectors(file, lstm_layers_count ,'float', 'W_c', W_c_list)
        append_vector_of_vectors(file, lstm_layers_count ,'float', 'W_o', W_o_list)
        append_vector_of_vectors(file, lstm_layers_count ,'float', 'U_i', U_i_list)
        append_vector_of_vectors(file, lstm_layers_count ,'float', 'U_f', U_f_list)
        append_vector_of_vectors(file, lstm_layers_count ,'float', 'U_c', U_c_list)
        append_vector_of_vectors(file, lstm_layers_count ,'float', 'U_o', U_o_list)
        append_vector_of_vectors(file, lstm_layers_count ,'float', 'b_i', b_i_list)
        append_vector_of_vectors(file, lstm_layers_count ,'float', 'b_f', b_f_list)
        append_vector_of_vectors(file, lstm_layers_count ,'float', 'b_c', b_c_list)
        append_vector_of_vectors(file, lstm_layers_count ,'float', 'b_o', b_o_list)

    file.close()

if __name__ == '__main__':
    model = load_model(MODEL_PATH)
    make_header(model)

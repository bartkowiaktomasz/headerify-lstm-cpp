import numpy as np
import pandas as pd
import keras

from keras.models import load_model
from kerasify import export_model

from config import *
from preprocessing import get_convoluted_data


def append_number_matrix(file, name, matrix):
    file.write("std::vector<std::vector<float>> " + name + " = {\n")
    for arr in matrix:
        file.write("{")
        file.write(', '.join(map(str, arr.tolist())))
        if not np.array_equal(arr, matrix[-1]):
            file.write("},\n")
        else:
            file.write("}\n")
    file.write("};\n\n")


def append_number_vector(file, name, vector):
    file.write("std::vector<float> " + name + " = {\n")
    file.write(', '.join(map(str, vector.tolist())))
    file.write("};\n\n")


def append_includes(file):
    file.write("#include <string>\n")
    file.write("#include <vector>\n")
    file.write("\n")

if __name__ == '__main__':

    # Load model
    model = load_model(MODEL_PATH)

    model_layers = [l for l in model.layers if type(l).__name__ not in ['Dropout']]
    num_layers = len(model_layers)

    with open("header.txt", "a") as file:
        append_includes(file)
        file.write("int NUM_LAYERS = " + str(num_layers) + ";\n")

        for layer in model_layers:
            layer_type = type(layer).__name__

            if layer_type == 'Dense':
                weights = layer.get_weights()[0]
                biases = layer.get_weights()[1]
                activation = layer.get_config()['activation']

                file.write("std::string DENSE_ACTIVATION = \"" + activation + "\";\n\n")
                append_number_matrix(file, "DENSE_WEIGHTS", weights)
                append_number_vector(file, "DENSE_BIASES", biases)


            elif layer_type == 'LSTM':
                recurrent_activation = layer.get_config()['recurrent_activation']
                activation = layer.get_config()['activation']
                return_sequences = int(layer.get_config()['return_sequences'])

                file.write("std::string LSTM_RECURRENT_ACTIVATION = \"" + recurrent_activation + "\";\n")
                file.write("std::string LSTM_ACTIVATION = \"" + activation + "\";\n")
                file.write("int RETURN_SEQUENCES = " + str(return_sequences) + ";\n")


                W = model.layers[0].get_weights()[0]
                U = model.layers[0].get_weights()[1]
                b = model.layers[0].get_weights()[2]
                num_units = int(int(model.layers[0].trainable_weights[0].shape[1])/4)
                file.write("int N_HIDDEN_NEURONS = " + str(num_units) + ";\n\n")

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

                append_number_matrix(file, "W_i", W_i)
                append_number_matrix(file, "W_f", W_f)
                append_number_matrix(file, "W_c", W_c)
                append_number_matrix(file, "W_o", W_o)
                append_number_matrix(file, "U_i", U_i)
                append_number_matrix(file, "U_f", U_f)
                append_number_matrix(file, "U_c", U_c)
                append_number_matrix(file, "U_o", U_o)

                append_number_vector(file, "b_i", b_i)
                append_number_vector(file, "b_f", b_f)
                append_number_vector(file, "b_c", b_c)
                append_number_vector(file, "b_o", b_o)


    # data = pd.read_pickle(DATA_PATH)
    # X_test, _ = get_convoluted_data(data)
    # print(model.predict(X_test[0].reshape(1,40,3)))

    # export_model(model, 'lstm.model')

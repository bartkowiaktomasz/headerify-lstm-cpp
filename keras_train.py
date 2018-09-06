"""
Script for training a LSTM network.
Usage:
Place your training data in DATA_PATH and run the script.
The script will build a model and save it to MODEL_PATH.
Paths can be edited in config file.
"""

from __future__ import print_function
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from preprocessing import get_convoluted_data
from config import *

def createLSTM(X_train,
               y_train,
               X_test,
               y_test):
    """
    Create a LSTM model.
    Take as input training and testing data an labels.
    Hyperparameters of the model are taken from the config file.
    """

    # Build an LSTM model
    # For more LSTM cells, add an argument
    # "return_sequences=True" in the first LSTM cell
    model = Sequential()
    model.add(LSTM(N_HIDDEN_NEURONS,
              input_shape=(SEGMENT_TIME_SIZE, N_FEATURES)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(N_CLASSES, activation='sigmoid'))
    adam_optimizer = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=N_EPOCHS,
              validation_data=[X_test, y_test])

    return model

if __name__ == '__main__':

    # Load data
    data = pd.read_pickle(DATA_PATH)
    data_convoluted, labels = get_convoluted_data(data)
    X_train, X_test, y_train, y_test = train_test_split(data_convoluted,
                                                        labels,test_size=TEST_SIZE,
                                                        random_state=RANDOM_SEED,
                                                        shuffle=True)
    # Build a model
    model = createLSTM(X_train, y_train, X_test, y_test)
    model.save(MODEL_PATH)

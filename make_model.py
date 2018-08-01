import numpy as np
import pandas as pd
import keras

from keras.models import load_model
from kerasify import export_model

from config import *
from preprocessing import get_convoluted_data

if __name__ == '__main__':

    # Load model
    model = load_model(MODEL_PATH)
    data = pd.read_pickle(DATA_PATH)
    X_test, _ = get_convoluted_data(data)
    print(model.predict(X_test[0].reshape(1,40,3)))

    # export_model(model, 'lstm.model')

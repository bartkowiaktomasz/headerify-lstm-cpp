# Headerify LSTM for C++
**Headerify LSTM for C++** is a small tool for coverting keras (1-layer LSTM + dense a.k.a fully connected layer) LSTM model to C++ headerfile that can be uploaded directly on the Arduino device allowing for on-board classification.

The tool is based on the amazing project by [Robert (moof2k) Wrose](https://github.com/moof2k) called [kerasify](https://github.com/moof2k/kerasify).

> Kerasify is a small library for running trained Keras models from a C++ application.

**Headerify LSTM for C++** is a modification of [kerasify](https://github.com/moof2k/kerasify) focusing solely on LSTM networks, allowing, i.e. for on-board [Activity recognition](https://github.com/bartkowiaktomasz/Fitness-Activity-Classification-with-LSTMs) based on IMU readings (data from the accelerometer/gyroscope/magnetometer). Unlike [kerasify](https://github.com/moof2k/kerasify) it allows for convering keras `.h5` directly to `.h` headerfile and then restoring the model and performing classification using this headerfile, without a need to use `file streams`. The tool was created with purpose to be used on the `Arduino` (or similar) devices, where reading from a file is impossible (at least on versions which do not support file access).

## Design features
 - Supports 2-layer model (1-layer LSTM + 1-layer Dense)
 - CPU only, no GPU
 - No external dependencies apart from STL for C++ code
 - Compatible with C++11
 - No file streams used (model stored as `.h` headerfile)
 - Supports the following activations: (Linear, ReLU, Softplus, Sigmoid, Tanh, Hard sigmoid)

## Dependencies (C++ code)

 - Standard Template Library (STL)
Lightweight version of STL library for Arduino can be found under the directory `ArduinoSTL`

 ## Dependencies (Python code)
 - **tensorflow**
 - scipy==1.1.0
 - numpy==1.14.5
 - pandas==0.23.1
 - Keras==2.2.0
 - scikit-learn==0.19.1
 - h5py==2.8.0

First you should install Tensorflow (code was tested using Tensorflow 1.8). Then you can install dependecies with:
`pip install -r requirements.txt`

# Usage
1. Clone the repository
`git clone https://github.com/bartkowiaktomasz/Headerify-lstm-cpp.git`
2. Place your keras LSTM `model.h5` in the `models/` directory.
3. Run `python make_header.py` to create the headerfile.
New file `LSTM_model.h` is created as a result.
4.  Test the model.
Check `test_cpp_model.cpp` file as an example.
~~~~
#include <iostream>
#include "keras_model.h"
#include "utils.h"
#include "config.h"

int main() {

    KerasModel model;		// Initialize model
    model.LoadModel();		// Load the LSTM model

    Tensor in(x, y);	// Instantiate a tensor of shape (x,y)
	/*
	 Put your data to be classified (x * y numbers separated
	 by comma) inbetween the curly brackets "{}". You can
	 "push_back" since in.data_ is a <float> vector.
	*/
    in.data_ = { };

    Tensor out;		// This tensor will store the output
    model.Apply(&in, &out);		// Perform the calculations
    std::cout << softmax_to_label(out.data_) << std::endl;
    return 0;
}
~~~~
5. Compile with:
`g++ --std=c++11 -Wall -O3 test_cpp_model.cpp keras_model.cpp utils.cpp config.cpp -o output
`
6. Check the output with `./output`

# Important notes
The output of the classification is a vector of numbers (after softmax). The function `softmax_to_label` (in `utils.cpp`) converts it into categorical values. These can be changed in the `config.cpp` file.

`keras_train.py` allows for building a keras LSTM model (the script saves the model in `models/` directory) based on the `data.pckl` file found in `data/` directory. Here `data.pckl` contains readings from the accelerometer `x-acceleration, y-acceleration, z-acceleration`. `keras_train.py` uses `config.py` with global variables that define how data is preprocessed in `preprocessing.py` and with all hyperparameters of the LSTM network (i.e. learning rate, number of neurons). In short, the data is preprocessed using *sliding window approach* and the network is then trained on those *windows* of data (each window consists of a series of samples and one label associated with it).

More information on how data gets preprocessed can be found in my project on [Fitness Activity Recognition with LSTMs](https://github.com/bartkowiaktomasz/Fitness-Activity-Classification-with-LSTMs).

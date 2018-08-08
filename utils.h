/*
Util functions.
*/


#ifndef UTILS_H_
#define UTILS_H_

#include <string>
#include <vector>

// Convert a vector of numbers (output of softmax) to a label
std::string softmax_to_label(std::vector<float> output);

#endif

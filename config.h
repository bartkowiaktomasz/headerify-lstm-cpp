/*
Config file with global constants.
*/

#ifndef CONFIG_H_
#define CONFIG_H_

#include <vector>
#include <string>

namespace config{
  // Labels (activities) that are considered by the model
  extern std::vector<std::string> LABELS_NAMES;

  // Number of features (3 if only acceleration is used)
  // 9 if acceleration + gyro + magnetometer readings are used
  extern int N_FEATURES;

  // Size of a sliding window (how many samples does it contain)
  extern int SEGMENT_TIME_SIZE;

  // Sliding window shift (number of samples)
  extern int TIME_STEP;
}

#endif

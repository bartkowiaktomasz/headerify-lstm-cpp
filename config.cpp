#include <vector>
#include <string>

#include "config.h"

namespace config{
  std::vector<std::string> LABELS_NAMES = {
     "Pushup",
     "Pushup_Incorrect",
     "Squat",
     "Situp",
     "Situp_Incorrect",
     "Jumping",
     "Lunge"
   };

   int SEGMENT_TIME_SIZE = 40;
   int TIME_STEP = 20;
   int N_FEATURES = 3;
}

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

   int N_FEATURES = 3;
}

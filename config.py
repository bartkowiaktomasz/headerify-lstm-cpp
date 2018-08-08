"""
Config file.
All other scripts import variables from this config
to their global namespace.
"""
##################################################
### GLOBAL VARIABLES
##################################################
COLUMN_NAMES = [
    'activity',
    'acc-x-axis',
    'acc-y-axis',
    'acc-z-axis',
    'gyro-x-axis',
    'gyro-y-axis',
    'gyro-z-axis',
    'mag-x-axis',
    'mag-y-axis',
    'mag-z-axis'
]

LABELS_NAMES = [
    'Pushup',
    'Pushup_Incorrect',
    'Squat',
    'Situp',
    'Situp_Incorrect',
    'Jumping',
    'Lunge'
]

# Data directories
DATA_PATH = 'data/data.pckl'

# Model directories
MODEL_PATH = 'models/model.h5'
MODEL_PATH_DIR = 'models/'

##################################################
### MODEL
##################################################
# Used for shuffling data
RANDOM_SEED = 13

# Model
N_CLASSES = len(LABELS_NAMES)
N_FEATURES = 3

# Hyperparameters
N_EPOCHS = 30
LEARNING_RATE = 0.0005
N_HIDDEN_NEURONS = 30
BATCH_SIZE = 30
DROPOUT_RATE = 0.2

##################################################
### DATA COLLECTION/PREPROCESSING
##################################################

# Data preprocessing
TIME_STEP = 5
SEGMENT_TIME_SIZE = 40

# Train/test proportion
TEST_SIZE = 0.2

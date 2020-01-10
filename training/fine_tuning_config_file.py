# Learning rate parameters
BASE_LR = 0.0001
EPOCH_DECAY = 10 # number of epochs after which the Learning rate is decayed exponentially.
DECAY_WEIGHT = 0.01 # factor by which the learning rate is reduced.


# DATASET INFO
NUM_CLASSES = 10 # set the number of classes in your dataset
DATA_PATH = '/work/lisabdunlap/bam/data/obj/'
TEST_DATA_PATH = '/work/lisabdunlap/bam/data/obj/'


# DATALOADER PROPERTIES
BATCH_SIZE = 125 # Set as high as possible. If you keep it too high, you'll get an out of memory error.

### GPU SETTINGS
CUDA_DEVICE = 2 # Enter device ID of your gpu if you want to run on gpu. Otherwise neglect.
GPU_MODE = 1 # set to 1 if want to run on gpu.


# SETTINGS FOR DISPLAYING ON TENSORBOARD
USE_TENSORBOARD = 0 #if you want to use tensorboard set this to 1.
TENSORBOARD_SERVER = "YOUR TENSORBOARD SERVER ADDRESS HERE" # If you set.
EXP_NAME = "fine_tuning_experiment" # if using tensorboard, enter name of experiment you want it to be displayed as.

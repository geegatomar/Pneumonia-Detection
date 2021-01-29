# import the necessary packages
from os import path



# define the base path to the pneumonia dataset
BASE_PATH = "datasets"

# define the number of classes (currently 2, if we decide to classify further lung diseases, this can be updated)
NUM_CLASSES = 2

# define the path to the output training, validation, and testing
TRAIN_PATH = "datasets/train"
TEST_PATH = "datasets/test"
VAL_PATH = "datasets/val"

# define the batch size
BATCH_SIZE = 32                 # need to tune this and check

# define the image width and height
IMG_HEIGHT = 300                # need to tune this and check
IMG_WIDTH = 300

# define the path to where output logs will be stored
OUTPUT_PATH = path.sep.join([BASE_PATH, "output"])

# import the necessary packages
import torch
import os

# base path of the dataset
DATASET_PATH = os.path.join("dataset", "train")

# define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "Raw")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "Labeled")

# define the test split
TEST_SPLIT = 0.10

# determine the device to be used for training and evaluation
DEVICE = "cuda" #if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# initialize learning rate, number of epochs to train for, and the batch size
INIT_LR = 0.0005 # era 0.0005 inainte
NUM_EPOCHS = 40 # era 50 inainte
BATCH_SIZE = 15 # era 1 inainte

# define the input image dimensions
INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256

# define threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory
BASE_OUTPUT = "output"

# define the path to the output serialized model, model training, plot and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "UNet.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
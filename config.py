import os
import main
import data
"""
BATCH_SIZE = 1
IMAGE_SIZE = 299
NUM_EPOCHS = 10
OUT_DIR = "results/"
MODEL_DIR = "model/"
DATA_DIR = "data/"
TRAIN_DIR = "train/"
TEST_DIR = "test/"
CLASSIFY_DIR = "model/"

LOG_DIR = "log/"

def set_cofig(FLAGS):
    global MODEL_DIR, BATCH_SIZE, NUM_EPOCHS, TRAIN_DIR, TEST_DIR, CLASSIFY_DIR, DATA_DIR

    DATA_DIR = FLAGS[6]
    CLASSIFY_DIR = FLAGS[5]
    BATCH_SIZE = FLAGS[4]
    NUM_EPOCHS = FLAGS[3]
    MODEL_DIR = FLAGS[2]
    TRAIN_DIR = FLAGS[1]
    TEST_DIR = FLAGS[0]
"""
DEFAULT_CLASSIFY_IMAGE_SIZE = 299
DEFAULT_COLORIZATION_IMAGE_SIZE = 320

class Config():

    def __init__(self, flagsar):    #Passed in the format : [data_dir, save_dir, batch_sizes, num_epochs]
        self.data_dir = flagsar[0]        #Directory of data (contains folders of images, for testing, training, or classifying)
        self.save_dir = flagsar[1]        #Directory to save files, this is where models will be
        self.batchsize = flagsar[2]
        self.num_epochs = flagsar[3]
        self.model_name = flagsar[4]

        #Look at files for testing / training folders
        filelist = os.listdir(self.data_dir)
        self.test_dir = None
        self.train_dir = None
        self.result_dir = None

        #check if directory contains training data
        if "Training" in filelist:
            self.train_dir = os.path.join(self.data_dir,"Training")
         #check if directory contains testing data
        if "Testing" in filelist:
            self.test_dir = os.path.join(self.data_dir,"Testing")
        if "Results" in filelist:
            self.result_dir = os.path.join(self.data_dir,"Results")
            
    

    
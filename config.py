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
        batch_sizes = flagsar[2]
        self.num_epochs = flagsar[3]

        #Look at files for testing / training folders
        filelist = os.listdir(self.data_dir)
        self.test_dir = None
        self.train_dir = None

        #check if directory contains training data
        if "Training" in filelist:
            self.train_dir = os.path.join(self.data_dir,"Training")
         #check if directory contains testing data
        if "Testing" in filelist:
            self.test_dir = os.path.join(self.data_dir,"Training")
            
        ###     Data    ###
        self.classifier_data = None
        self.train_data = None
        self.test_data = None

        ###     Classifying     ###
        # Classifying will be done for training samples so data_dir/Training
        if(self.train_dir != None):
            self.classifier_data = data.Data(self.train_dir, DEFAULT_CLASSIFY_IMAGE_SIZE, batch_sizes[0], None) 
            
        ###     Training        ###
        #Training samples are placed in data_dir/Training
        print("Train dir")
        print(self.train_dir)
        if( self.train_dir != None):
            self.train_data = data.Data(self.train_dir, DEFAULT_COLORIZATION_IMAGE_SIZE, batch_sizes[1], flagsar[3]) 
            print("EPOCHS")
            print(self.train_data.num_epochs)

        ###     Testing         ###
        #Testing samples are placed in data_dir/Testing
        if( self.test_data != None):
            self.test_data = data.Data(self.test_dir, DEFAULT_COLORIZATION_IMAGE_SIZE, 1, None)
            
        

    
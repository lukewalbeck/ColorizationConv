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
DEFAULT_DATA_DIRS = ["Colorize_Data","ColorData"]
DEFAULT_TRAINING_DIRS = ["Training", "TrainingData", "Training_Data"]
DEFAULT_TESTING_DIRS = ["Testing", "TestingData", "Testing_Data"]
DEFAULT_RESULT_DIRS = ["Results", "Colorize_Results", "Output", "Outputs"]

class Config():

    def __init__(self, flagsar):    #Passed in the format : [data_dir, save_dir, batch_sizes, num_epochs]
        self.data_dir = flagsar[0]        #Directory of data (contains folders of images, for testing, training, or classifying)
        self.save_dir = flagsar[1]        #Directory to save files, this is where models will be
        self.batchsize = flagsar[2]
        self.num_epochs = flagsar[3]
        self.model_name = flagsar[4]
        self.image_size = flagsar[5]

        #Try to find default data dir
        current_file_list = os.listdir(os.getcwd())
        for default_dir in DEFAULT_DATA_DIRS:
            if default_dir in current_file_list:
                self.data_dir = os.path.join( os.getcwd(), default_dir)
                break
            if default_dir.lower() in current_file_list:
                self.data_dir = os.path.join( os.getcwd(), default_dir.lower())
                break

        #Look at files for required folders
        if(self.data_dir != None):
        
            filelist = os.listdir(self.data_dir)
            self.test_dir = None
            self.train_dir = None
            self.result_dir = None

            #check if directory contains training data
            for training_dir in DEFAULT_TESTING_DIRS:
                if training_dir in filelist:
                    self.train_dir = os.path.join(self.data_dir, training_dir)
                    break
                if training_dir.lower() in filelist:
                    self.train_dir = os.path.join(self.data_dir, training_dir.lower())
                    break
            #check if directory contains testing data
            for testing_dir in DEFAULT_TESTING_DIRS:
                if testing_dir in filelist:
                    self.test_dir = os.path.join(self.data_dir,testing_dir)
                    break
                if testing_dir.lower() in filelist:
                    self.test_dir = os.path.join(self.data_dir,testing_dir.lower())
                    break
            #check for results directory
            for output_dir in DEFAULT_RESULT_DIRS:
                if output_dir in filelist:
                    self.result_dir = os.path.join(self.data_dir, output_dir)
                    break
                if output_dir.lower() in filelist:
                    self.result_dir = os.path.join(self.data_dir, output_dir.lower())
                    break

            
    

    
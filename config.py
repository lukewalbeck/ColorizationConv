import os
import main

BATCH_SIZE = 1
IMAGE_SIZE = 320
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
    
import os
import data
import config
import datetime
import argparse
import sys
import tensorflow as tf
import colornet

FLAGS = None

def main(_):
    print(config.MODEL_DIR)
    #FLAGS.image_dir
    # READ DATA
    #train_data = data.DATA(config.TRAIN_DIR)
    print("Train Data Loaded")
    # TRAIN MODEL
    #colornet.train(train_data)
    print("Model Trained")
    # TEST MODEL
    #test_data = data.DATA(config.TEST_DIR)
    print("Test Data Loaded")
    #colornet.test(test_data)
    print("Image Reconstruction Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_dir',
        type=str,
        default='',
        help='Path to folders of testing images.'
    )

    parser.add_argument(
        '--train_dir',
        type=str,
        default='',
        help='Path to folders of training images.'
    )

    parser.add_argument(
        '--model_dir',
        type=str,
        default='model/',
        help='Path to folder containing model.'
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        default=config.NUM_EPOCHS,
        help='Number of epochs to run.'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=config.BATCH_SIZE,
        help='Batch size.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


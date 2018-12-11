import os
import data
import config
import datetime
import argparse
import sys
import tensorflow as tf
import colornet
import config 
import classifier
import colornet

FLAGS = None

def main(_):
    #parse arguments from command line
    cf = config.Config([FLAGS.data_dir, FLAGS.save_dir, FLAGS.batch_size, FLAGS.num_epochs])

    # Verify can run
    if(FLAGS.train_net and cf.train_dir == None):
        print("Error: No training data")
        sys.exit(-1)
    if(FLAGS.test_net and cf.test_dir == None):
        print("Error: No testing data")
        sys.exit(-1)

    # Run net
    if(FLAGS.train_net):
        colornet.train(cf)

    if(FLAGS.test_net):
        colornet.test(cf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='Number of epochs to run.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=os.getcwd(),
        help='Directory to load images from, must contain a \'Training\' and \'Testing\' folder for training and testing data.'
    ) 
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Size of batch.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.getcwd(),
        help='Directory to output resulting images from colorization.'
    ) 
    parser.add_argument(
        '--model_name',
        type=str,
        default="colornet_model",
        help='Name of the model to save or load.'
    ) 
    parser.add_argument(
        '--train_net',
        type=bool,
        default=False,
        help='Train model on training samples'
    ) 
    parser.add_argument(
        '--test_net',
        type=bool,
        default=False,
        help='Test model on testing samples'
    ) 
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


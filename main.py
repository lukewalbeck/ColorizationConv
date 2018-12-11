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
    cf = config.Config([FLAGS.data_dir, FLAGS.save_dir, FLAGS.colorize_batch_size, FLAGS.num_epochs])
    colornet.run(cf)


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
        default= None,
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
        default=100,
        help='Number of epochs to run.'
    )

    parser.add_argument(
        '--classify_batch_size',
        type=int,
        default=1,
        help='Batch size for classification.'
    )
    parser.add_argument(
        '--colorize_batch_size',
        type=int,
        default=1,
        help='Batch size for colorization.'
    )
    parser.add_argument(
        '--classify_dir',
        type=str,
        default="",
        help='Classify directory.'
    )   
    parser.add_argument(
        '--data_dir',
        type=str,
        default="",
        help='Data directory.'
    ) 
    parser.add_argument(
        '--save_dir',
        type=str,
        default="",
        help='Save directory.'
    ) 
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


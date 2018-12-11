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

FLAGS = None

def main(_):
    #[data_dir, save_dir, batch_size, num_epochs]
    #config.set_cofig([FLAGS.test_dir, FLAGS.train_dir, FLAGS.model_dir, FLAGS.num_epochs, FLAGS.batch_size, FLAGS.classify_dir, FLAGS.data_dir])
    #train_data = data.DATA(config.TRAIN_DIR)
    batch_sizes = [FLAGS.classify_batch_size, FLAGS.colorize_batch_size]
    colorization_config = config.Config([FLAGS.data_dir, FLAGS.model_dir, batch_sizes, FLAGS.num_epochs])

    #classify 
    classifier.test(colorization_config, "final_graph.pb")
    print("\n\nFinished Classifying\n\n")

    #train colorization
    colornet.train(colorization_config)
    print("\n\nFinished Training\n\n")

    #test colorization
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
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


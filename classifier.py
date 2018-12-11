import tensorflow as tf
import config
import os
import numpy as np
import skimage.color as color
import skimage.io as io

def test(config, model_name):

    if(config.train_dir == None or config.classifier_data == None):
      print("Error: No training data to classify")
      sys.exit(-1)
    
    batch_size = config.classifier_data.batch_size
    image_size = config.classifier_data.image_size

    #load graph
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(os.path.join(config.save_dir, model_name), "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
        x = tf.placeholder(tf.float32, shape = [batch_size, image_size, image_size, 3], name = 'x')

    #classify sample
    with tf.Session(graph = graph) as session:
        total_batch = int(config.classifier_data.size/batch_size)
        for _ in range(total_batch):
            batchX, filelist = config.classifier_data.generate_stacked_grey()
            print(filelist) 
            output = session.run(graph.get_operation_by_name("import/final_result").outputs[0], 
                feed_dict = {graph.get_operation_by_name("import/Placeholder").outputs[0]: batchX})
            print(output)

"""
def test(data):

    #load graph
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(config.MODEL_DIR, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
        x = tf.placeholder(tf.float32, shape = [config.BATCH_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE, 3], name = 'x')

    #classify sample
    with tf.Session(graph = graph) as session:
        total_batch = int(data.size/config.BATCH_SIZE)
        for _ in range(total_batch):
            batchX, filelist = data.generate_stacked_grey()   
            print(filelist) 
            output = session.run(graph.get_operation_by_name("import/final_result").outputs[0], 
                feed_dict = {graph.get_operation_by_name("import/Placeholder").outputs[0]: batchX})
            print(output)
"""
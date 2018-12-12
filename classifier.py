import tensorflow as tf
import tensorflow.math
import config
import os
import numpy as np
import skimage.color as color
import skimage.io as io
import sys
import cv2
import data 

def classify(model_dir,model_name, classify_dir):

    image_size = 299
    batch_size = 1
    classify_data = data.Data(classify_dir, image_size, batch_size)

    if(classify_dir == None or classify_data == None):
      print("Error: No training data to classify")
      sys.exit(-1)

    #load labels exist
    labels = load_labels(os.path.join(model_dir, "labels.txt"))
    #load graph
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(os.path.join(model_dir, model_name), "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
        x = tf.placeholder(tf.float32, shape = [batch_size, image_size, image_size, 3], name = 'x')

    #classify sample
    outputs = []
    with tf.Session(graph = graph) as session:
        total_batch = int(classify_data.size/batch_size)
        for _ in range(total_batch):
            batchX, filelist = classify_data.generate_stacked_grey()
            #print(filelist) 
            output = session.run(graph.get_operation_by_name("import/final_result").outputs[0], 
                feed_dict = {graph.get_operation_by_name("import/Placeholder").outputs[0]: batchX})
            outputs.append(labels[find_label(output, labels)])
    return np.array(outputs)

def load_labels(label_file):
    label = []
    all_labels = tf.gfile.GFile(label_file).readlines()
    for l in all_labels:
        label.append(l.rstrip())
    return label
    
def find_label(output, labels):
    array = output[0]
    index = np.argmax(array)
    return index
   
       
            

"""
if __name__ == "__main__":
    output = test('D:\COMP380\Color\Model', 'final_graph.pb', 'D:\COMP380\Color\Colorize_Data\Testing' , 299, 1)           
    print(output[0][0])
    print(output[0][1])


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
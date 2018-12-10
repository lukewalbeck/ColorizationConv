import tensorflow as tf
import config
import numpy as np
import skimage.color as color
import skimage.io as io

def test(data):

    #load graph
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(config.MODEL_DIR, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
        x = tf.placeholder(tf.float32, shape = [config.BATCH_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE, 1], name = 'x')

    #classify sample
    with tf.Session(graph = graph) as session:
        total_batch = int(data.size/config.BATCH_SIZE)
        for _ in range(total_batch):
            batchX, batchY, filelist = data.generate_batch()   
            print(filelist) 
            output = session.run(graph.get_operation_by_name("import/final_result").outputs[0], 
                feed_dict = {graph.get_operation_by_name("import/Placeholder").outputs[0]: batchX})
            print(output)
    
import tensorflow as tf
import config

def test(data):
    x = tf.placeholder(tf.float32, shape = [None, 256, 256, 1], name = 'x')
    ytrue = tf.placeholder(tf.float32, shape = [None, 256, 256, 2], name = 'ytrue')
    saver = tf.train.Saver()
    with tf.Session() as session:
        #saver.restore(session, os.path.join(config.MODEL_DIR, "model" +
        #                                    str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))

        avg_cost = 0
        total_batch = int(data.size/config.BATCH_SIZE)
        for _ in range(total_batch):
            batchX, batchY, filelist = data.generate_batch()           
            output = session.run(tf.import_graph_def(config.MODEL_DIR), feed_dict = {x: batchX, ytrue: batchY})*128
            image = np.zeros([256, 256, 3])
            image[:,:,0]=batchX[0][:,:,0]
            image[:,:,1:]=output[0]
            image = color.lab2rgb(image)
            io.imsave("test.jpg", image)
import tensorflow as tf
import os
import config
import skimage.color as color
import skimage.io as io
import numpy as np



def train(data):
    x = tf.placeholder(tf.float32, shape = [None, 256, 256, 1], name = 'x')
    ytrue = tf.placeholder(tf.float32, shape = [None, 256, 256, 2], name = 'ytrue')
    loss = tf.losses.mean_squared_error(labels = ytrue, predictions = getColorization(x))
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
    saver = tf.train.Saver()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        print('All variables Initialized')
        for epoch in range(config.NUM_EPOCHS):
            avg_cost = 0
            for batch in range(int(data.size/config.BATCH_SIZE)):
                batchX, batchY, _ = data.generate_batch()
                feed_dict = {x: batchX, ytrue: batchY}
                loss_val = session.run(
                    optimizer, feed_dict=feed_dict)
                print("batch:", batch, " loss: ", loss_val)
                avg_cost += loss_val / int(data.size/config.BATCH_SIZE)
            print("Epoch:", (epoch + 1), "cost =",
                  "{:.5f}".format(avg_cost))

        save_path = saver.save(session, os.path.join(
            config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))
        print("Model saved in path: %s" % save_path)


def test(data):
    x = tf.placeholder(tf.float32, shape = [None, 256, 256, 1], name = 'x')
    ytrue = tf.placeholder(tf.float32, shape = [None, 256, 256, 2], name = 'ytrue')
    saver = tf.train.Saver()
    with tf.Session() as session:
        saver.restore(session, os.path.join(config.MODEL_DIR, "model" +
                                            str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))
        avg_cost = 0
        total_batch = int(data.size/config.BATCH_SIZE)
        for _ in range(total_batch):
            batchX, batchY, filelist = data.generate_batch()           
            output = session.run(getColorization(x), feed_dict = {x: batchX, ytrue: batchY})*128
            image = np.zeros([256, 256, 3])
            image[:,:,0]=batchX[0][:,:,0]
            image[:,:,1:]=output[0]
            image = color.lab2rgb(image)
            io.imsave("test.jpg", image)

def create_weights(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def create_bias(size):
  return tf.Variable(tf.constant(0.1, shape = [size]))

def convolution(inputs, num_channels, filter_size, num_filters):
  weights = create_weights(shape = [filter_size, filter_size, num_channels, num_filters])
  bias = create_bias(num_filters)
  
  ## convolutional layer
  layer = tf.nn.conv2d(input = inputs, filter = weights, strides= [1, 1, 1, 1], padding = 'SAME') + bias
  layer = tf.nn.tanh(layer)
  return layer

def getColorization(image):
    conv1 = convolution(image, 1, 3, 3)
    max1 = maxpool(conv1, 2, 2)
    conv2 = convolution(max1, 3, 3, 8)
    max2 = maxpool(conv2, 2, 2)
    conv3 = convolution(max2, 8, 3, 16)
    max3 = maxpool(conv3, 2, 2)
    conv4 = convolution(max3, 16, 3, 16)
    max4 = maxpool(conv4, 2, 2)
    conv5 = convolution(max4, 16, 3, 32)
    max5 = maxpool(conv5, 2, 2)
    conv6 = convolution(max5, 32, 3, 32)
    max6 = maxpool(conv6, 2, 2)
    conv7 = convolution(max6, 32, 3, 64)
    upsample1 = upsampling(conv7)
    conv8 = convolution(upsample1, 64, 3, 32)
    upsample2 = upsampling(conv8)
    conv9 = convolution(upsample2, 32, 3, 32)
    upsample3 = upsampling(conv9)
    conv10 = convolution(upsample3, 32, 3, 16)
    upsample4 = upsampling(conv10)
    conv11 = convolution(upsample4, 16, 3, 16)
    upsample5 = upsampling(conv11)
    conv12 = convolution(upsample5, 16, 3, 8)
    upsample6 = upsampling(conv12)
    conv13 = convolution(upsample6, 8, 3, 2)
    return conv13


def maxpool(inputs, kernel, stride):
  layer = tf.nn.max_pool(value = inputs, ksize = [1, kernel, kernel, 1], strides = [1, stride, stride, 1], padding = "SAME")
  return layer
  
def upsampling(inputs):
  layer = tf.image.resize_nearest_neighbor(inputs, (2*inputs.get_shape().as_list()[1], 2*inputs.get_shape().as_list()[2]))
  return layer



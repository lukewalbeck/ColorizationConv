import tensorflow as tf
import os
import config
import skimage.color as color
import skimage.io as io
import numpy as np
import sys
import cv2

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

def train(config):
  
    #verify training can run
    if(config.train_dir == None or config.train_data == None):
      print("Error: No training data")
      sys.exit(-1)

    #parameters
    num_epochs = config.train_data.num_epochs
    batch_size = config.train_data.batch_size
    img_size = config.train_data.image_size

    x = tf.placeholder(tf.float32, shape = [batch_size, img_size, img_size, 1], name = 'x')
    ytrue = tf.placeholder(tf.float32, shape = [batch_size, img_size, img_size, 2], name = 'ytrue')

    #construct model
    output = getColorization(x)

    #define loss
    #loss = tf.losses.mean_squared_error(labels = ytrue, predictions = output)
    #cost = tf.reduce_mean(tf.squared_difference(ytrue, output))
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=ytrue))
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss_op)

    correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(ytrue, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        print('All variables Initialized')
        for epoch in range(num_epochs):
            avg_cost = 0
            for batch in range(int(config.train_data.size/batch_size)):
                batchX, batchY, _ = config.train_data.generate_batch()
                session.run(optimizer, feed_dict={x: batchX, ytrue: batchY})
                if epoch % 10 == 0 or epoch == 1:
                  acc, loss = session.run([accuracy, loss_op], feed_dict={x: batchX, ytrue: batchY})                
                  print("Step " + str(epoch) + ", Minibatch Loss= " +  "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))

                #loss_val = session.run(optimizer, feed_dict={x: batchX, ytrue: batchY})
                #avg_cost += loss_val / int(data.size/config.BATCH_SIZE)
                #print("Epoch:", (epoch + 1), "cost =", "{:.5f}".format(avg_cost))


            save_path = saver.save(session, os.path.join(config.save_dir, "model" + str(batch_size) + "_" + str(num_epochs) + ".ckpt"))
        print("Model saved in path: %s" % save_path)
    return save_path


def test(config, model_name):
    img_size = config.test_data.image_size
    batch_size = config.test_data.batch_size
    num_epochs = config.test_data.num_epochs

    x = tf.placeholder(tf.float32, shape = [None, img_size, img_size, 1], name = 'x')
    ytrue = tf.placeholder(tf.float32, shape = [None, img_size, img_size, 2], name = 'ytrue')

    saver = tf.train.Saver()
    with tf.Session() as session:
        saver.restore(session, os.path.join(config.save_dir, "model" + str(batch_size) + "_" + str(num_epochs) + ".ckpt"))
        avg_cost = 0
        total_batch = int(config.test_data.size/batch_size)
        for _ in range(total_batch):
            batchX, batchY, filelist = config.test_data.generate_batch() 
            print(batchX)
            print(batchY)   
            output = session.run(getColorization(x), feed_dict = {x: batchX, ytrue: batchY})*128
            image = np.zeros([img_size, img_size, 3])
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


def maxpool(inputs, kernel, stride):
  layer = tf.nn.max_pool(value = inputs, ksize = [1, kernel, kernel, 1], strides = [1, stride, stride, 1], padding = "SAME")
  return layer
  
def upsampling(inputs):
  layer = tf.image.resize_nearest_neighbor(inputs, (2*inputs.get_shape().as_list()[1], 2*inputs.get_shape().as_list()[2]))
  return layer



import cv2
import matplotlib
import numpy as np
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import lab2rgb, rgb2lab
from skimage import io
import os
import config

def createModel():
    model = Sequential()
    model.add(InputLayer(input_shape=(None, None, 1)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3,3), activation='tanh', padding='same'))
    model.compile(optimizer='rmsprop', loss="mse")
    return model
    


def batch_create(src):
    INPUT_DIR = src
    input = []
    output = []
    for filename in os.listdir(INPUT_DIR):
        filesrc = os.path.join(INPUT_DIR, filename)
        image = img_to_array(load_img(filesrc, target_size=(200,200))) / 255
        lab_image = rgb2lab(image)
        lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]
        X = lab_image_norm[:,:,0] #bw image
        Y = lab_image_norm[:,:,1:] #color image
        X = X.reshape(X.shape[0], X.shape[1], 1)
        Y = Y.reshape(Y.shape[0], Y.shape[1], 2)
        input.append(X)
        output.append(Y)
 
    return np.array(input), np.array(output)

def run(config):
    training_src = 'Training/'
    nnmodel = createModel()
    print("Model created")
    input_training, output_training = batch_create(training_src)
    nnmodel.fit(x=input_training, y= output_training, batch_size=config.batch_size, epochs=config.num_epochs, verbose=1)
    nnmodel.evaluate(input_training, output_training, batch_size=config.batch_size)
    print("Training finished")

    testing_src = 'Testing/'
    testing, _ = batch_create(testing_src)
    
    output = nnmodel.predict(testing)
    print("Testing finished")
    cur = np.zeros((200, 200, 3))
    
    for i in range(len(output)):
        cur[:,:,0] = testing[i][:,:,0]
        cur[:,:,1:] = output[i]
        cur = (cur * [100, 255, 255]) - [0, 128, 128]
        rgb_image = lab2rgb(cur)
        result_src = 'results/'
        io.imsave(os.path.join(result_src, "result" + str(i) + ".png"), rgb_image)

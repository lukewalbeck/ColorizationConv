import cv2
import matplotlib
import numpy as np
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import lab2rgb, rgb2lab
import skimage.exposure
from skimage import io
import os
import config
import sys
import classifier

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
    
def batch_create(src, image_size):
    INPUT_DIR = src
    input = []
    output = []
    print(src)
    i = 0
    for filename in os.listdir(INPUT_DIR):
        filesrc = os.path.join(INPUT_DIR, filename)
        image = img_to_array(load_img(filesrc, target_size=(image_size,image_size))) / 255
        lab_image = rgb2lab(image)
        lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]
        X = lab_image_norm[:,:,0] #bw image
        Y = lab_image_norm[:,:,1:] #color image
        X = X.reshape( X.shape[0], X.shape[1], 1)
        Y = Y.reshape( Y.shape[0], Y.shape[1], 2)
        input.append(X)
        output.append(Y)
        i += 1
 
    return np.array(input), np.array(output)

def train(config):
    #net variables
    training_src = config.train_dir
    batch_size = config.batchsize
    num_epochs = config.num_epochs
    image_size = config.image_size

    #create model
    nnmodel = createModel()
    print("Model created")
    #get training and input data
    #classifications = classifier.classify('D:\COMP380\Color\Model', "final_graph.pb", training_src)
    input_training, output_training = batch_create(training_src, image_size)
    print(input_training.shape)
    print(output_training.shape)
    nnmodel.fit(x=input_training, y= output_training, batch_size=batch_size, epochs=num_epochs, verbose=1)
    nnmodel.evaluate(input_training, output_training, batch_size=batch_size)
    print("Training finished")

    # serialize model to JSON
    model_json_path = os.path.join(config.save_dir, config.model_name + ".json")
    model_json = nnmodel.to_json()

    with open(model_json_path, "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model_hdf5_path = os.path.join(config.save_dir,config.model_name + ".h5")
    nnmodel.save_weights(model_hdf5_path)
    print("Saved model to disk")

def load(config):
    """
    # load json and create model
    model_json_path = os.path.join(config.save_dir, config.model_name + ".json")
    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    """

    # load weights into new model
    loaded_model = createModel()
    model_hdf5_path = os.path.join(config.save_dir,config.model_name + ".h5") 
    loaded_model.load_weights(model_hdf5_path)
    print("Loaded model from drive")
    return loaded_model

def test(config):
    image_size = config.image_size
    nnmodel = load(config)
    #get testing data
    testing_src = config.test_dir
    testing, _ = batch_create(testing_src, image_size)
    #test model
    output = nnmodel.predict(testing)
    print("Testing finished")
    cur = np.zeros((image_size, image_size, 3))
    
    #save all results
    print("Saving Results")
    for i in range(len(output)):
        #convert from array
        cur[:,:,0] = testing[i][:,:,0]
        cur[:,:,1:] = output[i]
        cur = (cur * [100, 255, 255]) - [0, 128, 128]
        rgb_image = lab2rgb(cur)
        rgb_image = skimage.exposure.adjust_gamma(rgb_image, 1, 1)        
        #save result
        result_image_path = os.path.join(config.result_dir, "result" + str(i+1) + ".png")
        io.imsave(result_image_path, rgb_image)
        print("Saved result %d to: %s", i, result_image_path)

import numpy as np
import cv2
import os
import config


class DATA():

    def __init__(self, dirname):
        self.dir_path = os.path.join(config.DATA_DIR, dirname)
        self.filelist = os.listdir(self.dir_path)
        self.batch_size = config.BATCH_SIZE
        self.size = len(self.filelist)
        self.data_index = 0

    def read_img(self, filename):
        #print(filename)
        img = cv2.imread(filename, 3)
        height, width, channels = img.shape
        labimg = cv2.cvtColor(cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE)), cv2.COLOR_BGR2Lab)
       # print(img)
        return np.reshape(labimg[:,:,0], (config.IMAGE_SIZE, config.IMAGE_SIZE, 1)), labimg[:, :, 1:]


    def generate_batch(self):
        batch = []
        labels = []
        filelist = []
        for i in range(self.batch_size):
            filename = os.path.join(self.dir_path, self.filelist[self.data_index])
            filelist.append(self.filelist[self.data_index])
            greyimg, colorimg = self.read_img(filename)
            batch.append(greyimg)
            labels.append(colorimg)
            self.data_index = (self.data_index + 1) % self.size
        batch = np.asarray(batch)/config.IMAGE_SIZE
        labels = np.asarray(labels)/config.IMAGE_SIZE
        return batch, labels, filelist

    def generate_batch_color(self):
        batch = []
        filelist = []
        data_index = 0
        for i in range(self.batch_size):
            #get file
            filename = os.path.join(self.dir_path, self.filelist[self.data_index])
            filelist.append(self.filelist[self.data_index])
            #read file as rgb images
            colorimg = cv2.imread(filename, 3)
            #transform
            colorimg = cv2.cvtColor(cv2.resize(colorimg, (config.IMAGE_SIZE, config.IMAGE_SIZE)), cv2.COLOR_BGR2Lab)
            colorimg[:, :, 1:]
            #add to batch
            batch.append(colorimg)
            self.data_index = (self.data_index + 1) % self.size
        batch = np.asarray(batch)/255
        return batch, filelist
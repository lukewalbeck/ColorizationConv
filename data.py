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

    def get_resized_image(self, filename):
        img = cv2.imread(filename, 3)
        height, width, channels = img.shape
        resized_image = cv2.cvtColor(cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE)), cv2.COLOR_BGR2Lab)
        return resized_image

    def read_img(self, filename):
        #print(filename)
        #img = cv2.imread(filename, 3)
        #height, width, channels = img.shape
        #labimg = cv2.cvtColor(cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE)), cv2.COLOR_BGR2Lab)
        labimg = self.get_resized_image(filename)
        #print(img)
        grey_image = np.reshape(labimg[:,:,0], (config.IMAGE_SIZE, config.IMAGE_SIZE, 1))
        color_image = labimg[:, :, 1:]
        return grey_image, color_image


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

    def generate_stacked_grey(self):
        batch = []
        filelist = []
        for i in range(self.batch_size):
            filename = os.path.join(self.dir_path, self.filelist[self.data_index])
            filelist.append(self.filelist[self.data_index])
            #greyimg, _ = self.read_img(filename)
            img = self.get_resized_image(filename)
            stacked_grey_img = np.stack((img[:,:,0],)*3, axis=-1)
            print(stacked_grey_img.shape)
            batch.append(stacked_grey_img)
            self.data_index = (self.data_index + 1) % self.size
        batch = np.asarray(batch)/config.IMAGE_SIZE
        return batch,filelist
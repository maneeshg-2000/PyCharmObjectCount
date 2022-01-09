import random
import numpy as np
import cv2
import tensorflow as tf
import os
from PIL import Image
import json
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

# custom augmentation functions
#from image_augmentation import *
from common.Constants import *
npix = 224
target_size = (npix,npix,3)

def default_loader(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

#### tensorflow data generator
class TensorflowDataGenerator(tf.keras.utils.Sequence):  #
    def __init__(self, root, filenameList, batch_size, im_size=224, shuffle=True, transform=None,
                 target_transform=None, loader=default_loader):
        print("TensorflowDataGenerator: __init__")
        datafile = open(filenameList, "r")
        content_list = datafile.read().splitlines()

        random.seed(0)
        random.shuffle(content_list)  # shuffle it randomly
        self.imagesList = content_list

        self.root = root
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.im_size = im_size
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.ImageCount = len(self.imagesList)


    def on_epoch_end(self):
        print("TensorflowDataGenerator: on_epoch_end")
        if self.shuffle == True:
            random.shuffle(self.imagesList)

    def __len__(self):
        print("TensorflowDataGenerator: __len__")
        # Denotes the number of batches per epoch
        return len(self.imagesList) #// self.batch_size


    def __getitem__(self, index):
        print("TensorflowDataGenerator: __getitem__ ")
        # pick up the image
        filename = self.imagesList[index]
        metaFilename = os.path.join(METADATA_DIR, ('%s.json' % filename))
        imageFilename = os.path.join(IMAGE_DIR, ('%s.jpg' % filename))

        img = self.loader(imageFilename)
        #if self.transform is not None:
        #    img = self.transform(img)

        metadata = json.loads(open(metaFilename).read())
        quantity = metadata['EXPECTED_QUANTITY']
#        if self.target_transform is not None:
#            target = self.target_transform(target)
        return img, np.array([quantity])
"""
        # Generate one batch of data
        # Generate indices of the batch
        batch = self.train_imgs[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batch)
        return X, y
"""

"""
    def __get_data(self, batch):
        print("TensorflowDataGenerator:__get_data ")
        return

        X = []
        y = []
        for file_name in batch:
            im = cv2.imread(file_name, cv2.IMREAD_COLOR)
            im = augment(im, im_size=self.im_size,
                         random_crop_flag=1,
                         resize_flag=1)
            X.append(im)
            if 'dog' in file_name:
                y.append(1)
            elif 'cat' in file_name:
                y.append(0)
        return np.asarray(X), np.asarray(y).astype('float32').reshape((-1, 1))
"""

"""
    def load_val(self):
        print("TensorflowDataGenerator: load_val")

        X = []
        y = []
        for file_name in self.val_imgs:
            im = cv2.imread(file_name, cv2.IMREAD_COLOR)
            im = augment(im, self.im_size,
                         random_crop_flag=0,
                         resize_flag=1)
            X.append(im)
            if 'dog' in file_name:
                y.append(1)
            elif 'cat' in file_name:
                y.append(0)
        return (np.asarray(X), np.asarray(y).astype('float32').reshape((-1, 1)))
"""
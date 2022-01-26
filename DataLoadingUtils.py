import inspect
import json
import os
import random

import numpy as np
import tensorflow as tf
from keras.preprocessing import image

from common.Constants import *

model_dictionary = {m[0]:m[1] for m in inspect.getmembers(tf.keras.applications, inspect.ismodule)}

def default_loader(path,modelName):
    img = image.load_img(path, target_size=(224, 224))
    img1= tf.image.random_flip_left_right(img, 6)
    x = image.img_to_array(img1)
    #x = np.expand_dims(x, axis=0)
    x = model_dictionary[modelName.lower()].preprocess_input(x)
    return x


#### tensorflow data generator
class TensorflowDataGenerator(tf.keras.utils.Sequence):  #
    def __init__(self, root, filenameList, modelName, batch_size, im_size=224, shuffle=True, transform=None,
                 target_transform=None, loader=default_loader):
        datafile = open(filenameList, "r")
        content_list = datafile.read().splitlines()

        random.seed(0)
        random.shuffle(content_list)  # shuffle it randomly
        self.imagesList = content_list

        self.root = root
        self.modelName = modelName
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.im_size = im_size
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.ImageCount = len(self.imagesList)

    def on_epoch_end(self):
        if self.shuffle == True:
            random.shuffle(self.imagesList)

    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.imagesList) // self.batch_size

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indices of the batch
        batch = self.imagesList[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batch)
        return X, y

    def __get_data(self, batch):
        X = []
        y = []
        for filename in batch:
            metaFilename = os.path.join(METADATA_DIR, ('%s.json' % filename))
            imageFilename = os.path.join(IMAGE_DIR, ('%s.jpg' % filename))

            img = self.loader(imageFilename, self.modelName)

            X.append(img)

            metadata = json.loads(open(metaFilename).read())
            quantity = metadata['EXPECTED_QUANTITY']

            y.append(quantity)
        return np.asarray(X), np.asarray(y).astype('float32').reshape((-1,1))
import hashlib
import os
import platform
import shutil
import time
import json
import random
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from common.Constants import *

ClassTestFileName = [ TEST_SET_CLASS_BASED_FILENAME + "_Zero" + ".txt",
                 TEST_SET_CLASS_BASED_FILENAME + "_One" + ".txt",
                 TEST_SET_CLASS_BASED_FILENAME + "_Two" + ".txt",
                 TEST_SET_CLASS_BASED_FILENAME + "_Three" + ".txt",
                 TEST_SET_CLASS_BASED_FILENAME + "_Four" + ".txt",
                 TEST_SET_CLASS_BASED_FILENAME + "_Five" + ".txt",
                 TEST_SET_CLASS_BASED_FILENAME + "_Six" + ".txt"]

def dataVisualization():

    imageList = os.listdir(IMAGE_DIR)
    imageCount = len(imageList)
    #quantity_hist = np.zeros(500, dtype=int)


    if ((MAX_IMAGE_SET != "ALL")  and (imageCount > MAX_IMAGE_SET)):
        imageCount = MAX_IMAGE_SET * 2 #Reading twice the files of Max allowed set so that post filtering we get required data set
        print(imageCount)

    baseFilenameList = [i.split('.')[0] for i in imageList[:imageCount]]
    filterImageList = []

    for i in range(imageCount):
        metaFilename = os.path.join(METADATA_DIR, '%s.json' % baseFilenameList[i])
        imageFilename = os.path.join(IMAGE_DIR, '%s.jpg' % baseFilenameList[i])
        if (os.path.isfile(metaFilename) and os.path.isfile(imageFilename)) == False:
            print("Metadata File for Image %s does not exist", imageFilename)

        f = open(metaFilename)
        metadata = json.loads(f.read())
        f.close()
        quantity = metadata['EXPECTED_QUANTITY']
        binItemData = metadata["BIN_FCSKU_DATA"]
        itemTypeCount = len(binItemData)
        filterImageList.append([baseFilenameList[i], quantity, itemTypeCount])

    image_df = pd.DataFrame(filterImageList, columns=["filename", "Quantity", "ItemTypeCount"])
    image_df["Quantity"].hist(xlabelsize=6, ylabelsize=6)
    image_df["ItemTypeCount"].hist(xlabelsize=6, ylabelsize=6)

    print("The distribution of the number of file as per Quanity for each image:")
    print(Counter(image_df.Quantity.values))
    print("Most Common Item Count Distribution")
    print(Counter(image_df.Quantity.values).most_common(5))

    return image_df

def prepareTrainValidateTestSplitDataset(image_df, trainPercentage, validatePercentage, testPercentage):
    print("prepareTrainValidateTestSplitDataset -->")

    if ((1 - trainPercentage) != (validatePercentage + testPercentage)):
        print("Given Data Distribution is not correct")
        return;

    print("Image Count before Filtering ", image_df.shape[0])
    image_df = image_df[image_df['Quantity'] <= MAX_ITEM_COUNT]
    print("Image Count After Filtering based on Item Count", image_df.shape[0])
    image_df = image_df[image_df['ItemTypeCount'] <= MAX_ITEM_TYPE_COUNT]
    print("Image Count After Filtering based on Item Type Count", image_df.shape[0])

    imageCount = image_df.shape[0]

    if ((MAX_IMAGE_SET != "ALL")  and (imageCount > MAX_IMAGE_SET)):
        imageCount = MAX_IMAGE_SET
        image_df = image_df.iloc[:imageCount]
        print("Image Count After Image Limit Count check ", image_df.shape[0])

    testDF, vaidationDF, trainDF = np.split(image_df,
                                            [int(validatePercentage*len(image_df)),
                                             int(testPercentage*len(image_df))])

    # writing out to textfile
    os.makedirs(INTERMEDIATE_DIS, exist_ok=True)
    trainDF['filename'].to_csv(TRAIN_SET_FILENAME, sep=' ', index=False, header=False)
    vaidationDF['filename'].to_csv(VALIDATION_SET_FILENAME, sep=' ', index=False, header=False)
    testDF['filename'].to_csv(TEST_SET_FILENAME, sep=' ', index=False, header=False)

    for i in range(len(ClassTestFileName)):
        tempDF = testDF[testDF['Quantity'] == i]
        tempDF['filename'].to_csv(ClassTestFileName[i], sep=' ', index=False, header=False)

import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.Constants import *


def dataVisualization():

    imageList = os.listdir(IMAGE_DIR)
    imageCount = len(imageList)
    #quantity_hist = np.zeros(500, dtype=int)
    print("DIR Listing Complete %d" %(imageCount))

    if ((MAX_TRAIN_IMAGE_COUNT != "ALL")  and (imageCount > MAX_TRAIN_IMAGE_COUNT)):
        imageCount = MAX_TRAIN_IMAGE_COUNT * 2 #Reading twice the files of Max allowed set so that post filtering we get required data set
        print(imageCount)

    baseFilenameList = [ i.split('.')[0] for i in imageList[:imageCount]]
    filterImageList = []

    for i in range(imageCount):
        if int(i%1000)==0:
            print("Files Processed %d/%d" %(i,imageCount))
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
        #filterImageList.append([baseFilenameList[i], quantity, itemTypeCount])
        filterImageList.append([baseFilenameList[i], quantity])

    image_df = pd.DataFrame(filterImageList, columns=["filename", "Quantity"])
    image_df["Quantity"].hist(xlabelsize=6, ylabelsize=6)
    plt.savefig(INTERMEDIATE_DIR+"ImageDistribution.jpg")
    plt.show(block=False)
    plt.close()


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

    if ((MAX_TRAIN_IMAGE_COUNT != "ALL")  and (imageCount > MAX_TRAIN_IMAGE_COUNT)):
        imageCount = MAX_TRAIN_IMAGE_COUNT
        image_df = image_df.iloc[:imageCount]
        print("Image Count After Image Limit Count check ", image_df.shape[0])

    testDF, vaidationDF, trainDF = np.split(image_df,
                                            [int(validatePercentage*len(image_df)),
                                             int(testPercentage*len(image_df))])

    # writing out to textfile
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    trainDF['filename'].to_csv(TRAIN_SET_FILENAME, sep=' ', index=False, header=False)
    vaidationDF['filename'].to_csv(VALIDATION_SET_FILENAME, sep=' ', index=False, header=False)
    testDF['filename'].to_csv(TEST_SET_FILENAME, sep=' ', index=False, header=False)

    for i in range(len(ClassTestFileName)):
        tempDF = testDF[testDF['Quantity'] == i]
        tempDF['filename'].to_csv((TEST_SET_CLASS_BASED_FILENAME + ClassTestFileName[i] + ".txt"), sep=' ', index=False, header=False)

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
def plthist(dfsub, topn=-1):
    plt.figure(figsize=(5,2))
    plt.xlabel('Quantity in a bin image')
    plt.ylabel('The number of bin images')
    #plt.bar(dfsub.index, dfsub["Quantity"])
    plt.plot(dfsub[:topn])
    plt.yticks(fontsize=12)
    #plt.xticks(dfsub.index,dfsub["word"],rotation=90,fontsize=20)
    plt.title("Bin Item Count Distribution",fontsize=12)
    plt.show()


def dataVisualization():
    imageList = os.listdir(IMAGE_DIR)
    imageCount = len(imageList)
    quantity_hist = np.zeros(100, dtype=int)


    if ((MAX_IMAGE_SET != "ALL")  and (imageCount > MAX_IMAGE_SET)):
        imageCount = MAX_IMAGE_SET
        print(imageCount)
    baseFilenameList = [i.split('.')[0] for i in imageList[:imageCount]]
    filterImageList = []

    for i in range(imageCount):
        metaFilename = os.path.join(METADATA_DIR, '%s.json' % baseFilenameList[i])
        imageFilename = os.path.join(IMAGE_DIR, '%s.jpg' % baseFilenameList[i])
        if (os.path.isfile(metaFilename) and os.path.isfile(imageFilename) == False) :
            print("Metadata File for Image %s does not exist", imageFilename)

        metadata = json.loads(open(metaFilename).read())
        quantity = metadata['EXPECTED_QUANTITY']
        binItemData = metadata["BIN_FCSKU_DATA"]
        itemTypeCount = len(binItemData)
        filterImageList.append([baseFilenameList[i], quantity, itemTypeCount])
        quantity_hist[quantity] = quantity_hist[quantity] + 1

    print(len(filterImageList))

    image_df = pd.DataFrame(filterImageList, columns=["filename", "Quantity", "ItemTypeCount"])

    uni_filenames = np.unique(image_df.filename.values)
    print("The number of unique file names : {}".format(len(uni_filenames)))
    print("The distribution of the number of file as per Quanity for each image:")
    print(Counter(image_df.Quantity.values))
    print("Most Common Item Count Distribution")
    print(Counter(image_df.Quantity.values).most_common(5))

    #plthist(quantity_hist)
    #plthist(quantity_hist,10)
    return image_df

def prepareTrainValidateTestSplitDataset(image_df, trainPercentage, validatePercentage, testPercentage):
    print("prepareTrainValidateTestSplitDataset -->")
    print(ClassTestFileName)

    if ((1 - trainPercentage) != (validatePercentage + testPercentage)):
        print("Given Data Distribution is not correct")
        return;

    print("Image Count before Filtering ", image_df.shape[0])
    image_df = image_df[image_df['Quantity'] <= MAX_ITEM_COUNT]
    print("Image Count After Filtering basd on Item Count", image_df.shape[0])
    image_df = image_df[image_df['ItemTypeCount'] <= MAX_ITEM_TYPE_COUNT]
    print("Image Count After Filtering basd on Item Type Count", image_df.shape[0])

    imageList = os.listdir(IMAGE_DIR)
    imageCount = len(imageList)

    if ((MAX_IMAGE_SET != "ALL")  and (imageCount > MAX_IMAGE_SET)):
        imageCount = MAX_IMAGE_SET
        print(imageCount)

    baseFilenameList = [i.split('.')[0] for i in imageList[:imageCount]]

    listRandom = baseFilenameList.copy()
    random.shuffle(listRandom)

    # finding images that metadata exists
    metaAvail = np.zeros(imageCount, dtype=bool)
    for i in range(imageCount):
        metaFilename = os.path.join(METADATA_DIR, '%s.json' % baseFilenameList[i])
        imageFilename = os.path.join(IMAGE_DIR, '%s.jpg' % baseFilenameList[i])
        if os.path.isfile(metaFilename) and os.path.isfile(imageFilename) :
            # Filter Images with more than MAX_ITEM_COUNT
            metadata = json.loads(open(metaFilename).read())
            quantity = metadata['EXPECTED_QUANTITY']
            if (quantity <= MAX_ITEM_COUNT):
                metaAvail[i] = True
            #else:
                #print("Image %s is having % items in BIN, so skipping", baseFilenameList[i], quantity)



    # assign validataion & test set
    valSet = np.zeros(imageCount, dtype=bool)
    valSetImageCount = int(round(imageCount * validatePercentage))

    testSet = np.zeros(imageCount, dtype=bool)
    testSetImageCount = int(round(imageCount * testPercentage))

    count = 0
    for i in range(imageCount):
        idx = baseFilenameList.index(listRandom[i])
        if metaAvail[idx]:
            valSet[idx] = True
            count = count + 1
            if count == valSetImageCount:
                break

    count = 0
    random.shuffle(listRandom)
    for i in range(imageCount):
        idx = baseFilenameList.index(listRandom[i])
        if metaAvail[idx] and valSet[idx] != True:
            testSet[idx] = True
            count = count + 1
            if count == testSetImageCount:
                break

    # writing out to textfile
    os.makedirs(INTERMEDIATE_DIS, exist_ok=True)
    trainSetFilename = open(TRAIN_SET_FILENAME, 'w')
    testSetFilename = open(TEST_SET_FILENAME, 'w')
    validationSetFilename = open(VALIDATION_SET_FILENAME, 'w')
    for i in range(imageCount):
        if metaAvail[i]:
            if valSet[i]:
                validationSetFilename.write("%s\n" % (baseFilenameList[i]))
            elif testSet[i]:
                testSetFilename.write("%s\n" % (baseFilenameList[i]))

            else:
                trainSetFilename.write("%s\n" % (baseFilenameList[i]))
    trainSetFilename.close()
    testSetFilename.close()
    validationSetFilename.close()
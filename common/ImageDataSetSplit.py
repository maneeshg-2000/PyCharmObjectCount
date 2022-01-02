import hashlib
import os
import platform
import shutil
import time
import json
import random
import numpy as np

from common.Constants import *

def prepareTrainValidateTestSplitDataset(trainPercentage, validatePercentage, testPercentage):
    print("prepareTrainValidateTestSplitDataset -->")

    if ((1 - trainPercentage) != (validatePercentage + testPercentage)):
        print("Given Data Distribution is not correct")
        return;

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
            metaAvail[i] = True

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
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

    listRandom = [*range(imageCount)]
    random.shuffle(listRandom)

    # finding images that metadata exists
    metaAvail = np.zeros(imageCount, dtype=bool)
    for i in range(imageCount):
        metaFilename = os.path.join(METADATA_DIR, ('%05d.json' % (i + 1)))
        imageFilename = os.path.join(IMAGE_DIR, ('%05d.jpg' % (i + 1)))
        if os.path.isfile(metaFilename) and os.path.isfile(imageFilename) :
            metaAvail[i] = True

    # assign validataion & test set
    valSet = np.zeros(imageCount, dtype=bool)
    valSetImageCount = int(round(imageCount * validatePercentage))

    testSet = np.zeros(imageCount, dtype=bool)
    testSetImageCount = int(round(imageCount * testPercentage))

    count = 0
    random.shuffle(listRandom)
    for i in range(imageCount):
        idx = listRandom[i]
        if metaAvail[idx]:
            valSet[idx] = True
            count = count + 1
            if count == valSetImageCount:
                break

    count = 0
    random.shuffle(listRandom)
    for i in range(imageCount):
        idx = listRandom[i]
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
        if(i == 1201):
            print("1201", metaAvail[i])
        if metaAvail[i]:
            if valSet[i]:
                validationSetFilename.write("%05d\n" % (i + 1))
            elif testSet[i]:
                testSetFilename.write("%05d\n" % (i + 1))
            else:
                trainSetFilename.write("%05d\n" % (i + 1))
    trainSetFilename.close()
    testSetFilename.close()
    validationSetFilename.close()

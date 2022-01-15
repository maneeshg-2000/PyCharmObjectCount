from common.ImageFileIO import *

def readTrainData():
    datafile = open(TRAIN_SET_FILENAME, "r")
    content_list = datafile.read().splitlines()
    trainData = []

    trainData.append(["filename","Quanity"])
    for filename in content_list:
        trainData.append(readFileWithJson(filename))
    return trainData

def readTestData():
    datafile = open(TEST_SET_FILENAME, "r")
    content_list = datafile.read().splitlines()
    testData = []

    testData.append(["filename", "Quanity"])
    for filename in content_list:
        testData.append(readFileWithJson(filename))
    return testData

def readValidationData():
    datafile = open(VALIDATION_SET_FILENAME, "r")
    content_list = datafile.read().splitlines()
    validationData = []

    validationData.append(["filename", "Quanity"])
    for filename in content_list:
        validationData.append(readFileWithJson(filename))
    return validationData

from common.Constants import *
from common.ImageFileIO import *

def readTrainData():
    datafile = open(TRAIN_SET_FILENAME, "r")
    content_list = datafile.read().splitlines()

    for filename in content_list:
        readFileWithJson(filename)

def readTestData():
    print("readTestData")

def readValidationData():
    print("readValidationData")

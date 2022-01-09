# Imports
import argparse
import inspect
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from DataLoadingUtils import TensorflowDataGenerator
import tensorflow_datasets as tfds
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor

from common.ImageDataSetSplit import *
from ImageModule.CNNModule import *
from ImageModule.ResnetModule import *
from ImageModule.SiasmeVGGModule import *


model_dictionary = {m[0]:m[1] for m in inspect.getmembers(tf.keras.applications, inspect.isfunction)}
model_names = model_dictionary.keys()

parser = argparse.ArgumentParser(description='Image Object Counting Model')
parser.add_argument('--model', '-m', metavar='MODEL', default='VGG16', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: VGG16)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lrd','--learning-rate-decay-step', default=10, type=int, metavar='N', help='learning rate decay epoch')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--max-target', default=20, type=int, metavar='N', help='maximum number of target images for validation (default: 20)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', default=False, type=bool, metavar='BOOL', help='evaluate or train')

def compareAllAlgo():
    print("compareAllAlgo -->")
    countWithCNN()
    countWithCNNAndPreprocess()
    countWithResNet()
    countWithResNetAndPreprocess()
    countWithSiasme()
    countWithSiasmeAndPreprocess()

def show_choices():
    print('\nMenu')
    print('1. Count with CNN')
    print('2. Count with ResNet')
    print('3. Count with Siasme')
    print('4. Count with CNN and PreProcess')
    print('5. Count with ResNet and PreProcess')
    print('6. Count with Siasme and PreProcess')
    print('7. Print Comparison')
    print('p. Preprocess Input Images')
    print('X. Exit')

def oneTimeImageProcssing():
    # Resize all images to 224x244, One time operation, so commenting call
    #resizeImages()˳

    # Convert RGB to HSV Images
    #convertRGB2HSVImages()
    return


def mainMenu():
    global args
    args = parser.parse_args()

    # create model
    print("=> creating model '{}'".format(args.model))

    model_func = model_dictionary[args.model]
    print("Building Model")
    model = model_func()

    model.compile('adam', loss='mse')
    prepareTrainValidateTestSplitDataset(0.7,0.1,0.2)

    training_generator = TensorflowDataGenerator(IMAGE_DIR,TRAIN_SET_FILENAME,32)
    validation_generator = TensorflowDataGenerator(IMAGE_DIR,VALIDATION_SET_FILENAME,32)

    model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=12)

    """
    trainData = readTrainData()
    validationData = readValidationData()
    testData = readTestData()
    print ("Train Data", trainData)
    print("Validation Data", validationData)
    print("testData", testData)


    while(True):
        show_choices()
        choice = input('Enter choice(1-7,p,X): ')
        if choice == '1':
            countWithCNN()
            print("Count with CNN")
        elif choice == '2':
            countWithResNet()
            print("Count with ResNet")
        elif choice == '3':
            countWithSiasme()
            print("Count with Siasme")
        elif choice == '4':
            countWithCNNAndPreprocess()
            print("Count with CNN and PreProcess")
        elif choice == '5':
            countWithResNetAndPreprocess()
            print("Count with ResNet and PreProcess")
        elif choice == '6':
            countWithSiasmeAndPreprocess()
            print("Count with Siasme and PreProcess")
        elif choice == '7':
            compareAllAlgo()
            print("Show Comparison")
        elif choice == 'p':
            oneTimeImageProcessing()
            print("Preprocess Images")
        else:
            print('Invalid input')
            break
"""

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mainMenu()

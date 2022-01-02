from common.ImageDataSetSplit import prepareTrainValidateTestSplitDataset
from common.readDataset import readTrainData
from common.readDataset import readValidationData
from common.readDataset import readTestData

from ImageModule.CNNModule import *
from ImageModule.ResnetModule import *
from ImageModule.SiasmeModule import *

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
    print('X. Exit')


def mainMenu():

    prepareTrainValidateTestSplitDataset(0.7,0.1,0.2)
    trainData = readTrainData()
    validationData = readValidationData()
    testData = readTestData()
    print ("Train Data", trainData)
    print("Validation Data", validationData)
    print("testData", testData)

    while(True):
        show_choices()
        choice = input('Enter choice(1-7,X): ')
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
        else:
            print('Invalid input')
            break

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mainMenu()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

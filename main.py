# Imports
import argparse
import inspect
import tensorflow as tf
import keras
from DataLoadingUtils import TensorflowDataGenerator
from common.ImageDataSetSplit import *

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

model_dictionary = {m[0]:m[1] for m in inspect.getmembers(tf.keras.applications, inspect.isfunction)}
model_names = model_dictionary.keys()

parser = argparse.ArgumentParser(description='Image Object Counting Model')
parser.add_argument('--model', '-m', metavar='MODEL', default='ResNet50', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: ResNet50)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=3, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lrd','--learning-rate-decay-step', default=10, type=int, metavar='N', help='learning rate decay epoch')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--max-target', default=20, type=int, metavar='N', help='maximum number of target images for validation (default: 20)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', default=False, type=bool, metavar='BOOL', help='evaluate or train')


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

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.lr,
        decay_steps=10000,
        decay_rate=args.lrd)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=args.momentum)

    model.compile(loss='mse',optimizer=optimizer, metrics=['accuracy', 'mse'])

#    prepareTrainValidateTestSplitDataset(0.7,0.1,0.2)

    training_generator = TensorflowDataGenerator(IMAGE_DIR,TRAIN_SET_FILENAME,args.model,args.batch_size)
    validation_generator = TensorflowDataGenerator(IMAGE_DIR,VALIDATION_SET_FILENAME,args.model,args.batch_size)

    model.fit(x=training_generator,
                validation_data=validation_generator,
                epochs=args.epochs,
                verbose=1,
                use_multiprocessing=True,
                workers=12)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mainMenu()

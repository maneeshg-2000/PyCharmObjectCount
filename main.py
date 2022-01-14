# Imports
import argparse
import inspect
import tensorflow as tf
import keras
from DataLoadingUtils import TensorflowDataGenerator
from common.ImageDataSetSplit import *
from keras.callbacks import ModelCheckpoint
from datetime import date
from keras.utils.vis_utils import plot_model
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Flatten, Dense
import pandas as pd
from keras.callbacks import CSVLogger
import sklearn
from sklearn.metrics import ConfusionMatrixDisplay


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
parser.add_argument('--model', '-m', metavar='MODEL', default='VGG16', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: VGG16)')
parser.add_argument('--epochs', default=3, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=4, type=int, metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lrd','--learning-rate-decay-step', default=10, type=int, metavar='N', help='learning rate decay epoch')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--max-target', default=20, type=int, metavar='N', help='smaximum number of target images for validation (default: 20)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', default=False, type=bool, metavar='BOOL', help='evaluate or train')
parser.add_argument('--pd','--prepare-data', action='store_true', help='Prepare Data for Model Evaluation')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N', help='number of data loading workers (default: 12)')

def oneTimeImageProcssing():
    # Resize all images to 224x244, One time operation, so commenting call
    #resizeImages()˳

    # Convert RGB to HSV Images
    #convertRGB2HSVImages()
    return

model_dictionary_modules = {m[0]:m[1] for m in inspect.getmembers(tf.keras.applications, inspect.ismodule)}


def predictTestDataSet(modelname, model, filenameList):
    datafile = open(filenameList, "r")
    content_list = datafile.read().splitlines()
    testResult = []

    for filename in content_list:
        metaFilename = os.path.join(METADATA_DIR, ('%s.json' % filename))
        imageFilename = os.path.join(IMAGE_DIR, ('%s.jpg' % filename))

        img = image.load_img(imageFilename, target_size=(224, 224))
        img_x = image.img_to_array(img)
        img_x = np.expand_dims(img_x, axis=0)
        img_x = model_dictionary_modules[modelname.lower()].preprocess_input(img_x)

        metadata = json.loads(open(metaFilename).read())
        actualQuantity = metadata['EXPECTED_QUANTITY']

        predictedQuantity = model.predict(img_x)
        predictedQuantity = np.argmax(predictedQuantity)
        testResult.append([filename, actualQuantity, predictedQuantity])

    testResultDF = pd.DataFrame(testResult, columns=["filename", "ActualQuantity", "PredictedQuantity"])
    return testResultDF

def plotModelEvalMetric(modelName, trainMetricHistory, validationMetricHistory, metricName):
    # plotting training and validation metric
    #loss = history[metric]
    #val_loss = history['val_' + metric]
    epochs = range(1, len(trainMetricHistory) + 1)
    plt.plot(epochs, trainMetricHistory, color='red', label='Training ' + metricName )
    plt.plot(epochs, validationMetricHistory, color='green', label='Validation' + metricName)
    plt.title('Training and Validation '+ metricName)
    plt.xlabel('Epochs')
    plt.ylabel(metricName)
    plt.legend()
    plt.savefig(INTERMEDIATE_DIR+modelName+metricName+".jpg")
    plt.show()

def mainMenu():
    global args

    args = parser.parse_args()
    print(args)

    image_df = dataVisualization()

    if(args.pd):
        print("Calling Prepare Data")
        prepareTrainValidateTestSplitDataset(image_df, 0.7,0.1,0.2)

    # create model
    print("=> creating model '{}'".format(args.model))
    model_func = model_dictionary[args.model]

    print("Building Model")
    base_model = model_func(include_top=True,weights=None)
    base_model.summary()

    # Add a layer where input is the output of the  second last layer
    x = tf.keras.layers.Dense(8, activation='softmax', name='predictions')(base_model.layers[-2].output)

    # Then create the corresponding model
    my_model = Model(inputs=base_model.input, outputs=x)
    my_model.summary()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.lr,
        decay_steps=10000,
        decay_rate=args.lrd)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=args.momentum)


    my_model.compile(loss='mse',optimizer=optimizer, metrics=['accuracy', 'mse'])

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            my_model = keras.models.load_model(args.resume)
            filename, file_extension = os.path.splitext(args.resume)
            filename = filename + "_trainHistory.csv"
            if os.path.isfile(args.resume):
                train_history = pd.read_csv(filename, sep=',', engine='python')
                print(train_history)
            else:
                print("training history is not available")

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return
    else:
        # checkpoint
        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(patience=2),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("./",date.today().strftime("%m-%d-%Y"),'model.{epoch:02d}-{mse:.2f}.h5'),
                monitor='mse',
                verbose=1,
                save_best_only=True,
                mode='max'),
            tf.keras.callbacks.TensorBoard(log_dir='./logs'),
            CSVLogger((INTERMEDIATE_DIR + args.model +"_trainHistory.csv"), separator=',', append=False)
        ]

        training_generator = TensorflowDataGenerator(IMAGE_DIR,TRAIN_SET_FILENAME,args.model,args.batch_size)
        validation_generator = TensorflowDataGenerator(IMAGE_DIR,VALIDATION_SET_FILENAME,args.model,args.batch_size)

        hist = my_model.fit(x=training_generator,
                    validation_data=validation_generator,
                    epochs=args.epochs,
                    verbose=1,
                    use_multiprocessing=True,
                    callbacks=callbacks_list,
                    initial_epoch= args.start_epoch,
                    workers=args.workers)
        train_history = hist.history

        print('Final training loss \t', str(round((hist.history['loss'][-1]),2))+ ' %')
        print('Final Training accuracy ', str(round((hist.history['accuracy'][-1]*100),2))+ ' %')
        print('Final Validation loss \t', str(round((hist.history['val_loss'][-1]),2))+ ' %')
        print('Final Validation accuracy ', str(round((hist.history['val_accuracy'][-1]*100),2))+ '%')

        os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
        my_model.save((INTERMEDIATE_DIR + args.model +".h5"))


    plotModelEvalMetric(args.model,train_history["loss"],train_history["val_loss"], 'loss')
    plotModelEvalMetric(args.model,train_history["accuracy"],train_history["val_accuracy"], 'accuracy')
    plotModelEvalMetric(args.model,train_history["mse"],train_history["val_mse"], 'mse')

    testResultDF = predictTestDataSet(args.model, my_model, TEST_SET_FILENAME)
    testResultDF.to_csv(INTERMEDIATE_DIR +"allClassResult.csv", sep=',', index=False, header=True)


    ConfusionMatrixDisplay.from_predictions(testResultDF["ActualQuantity"], testResultDF["PredictedQuantity"])
    plt.savefig(INTERMEDIATE_DIR+args.model+"AllClassResultConfMetric.jpg")

    for i in range(len(ClassTestFileName)):
        testResultDF = predictTestDataSet(args.model, my_model,
                                          (TEST_SET_CLASS_BASED_FILENAME + ClassTestFileName[i] + ".txt"))
        testResultDF.to_csv((TEST_SET_CLASS_BASED_FILENAME + ClassTestFileName[i] + "result.csv"), sep=',', index=False,
                            header=True)
        if(testResultDF.shape[0]):
            ConfusionMatrixDisplay.from_predictions(testResultDF["ActualQuantity"], testResultDF["PredictedQuantity"])
            plt.savefig(INTERMEDIATE_DIR + args.model + ClassTestFileName[i]+"ConfMetric.jpg")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mainMenu()

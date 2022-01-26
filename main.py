# Imports
import argparse
import inspect
import os.path

import keras
import tensorflow as tf
from keras.models import Model
from keras.preprocessing import image
from keras.utils.vis_utils import plot_model
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from common.ImageDataSetSplit import *

model_dictionary = {m[0]:m[1] for m in inspect.getmembers(tf.keras.applications, inspect.isfunction)}
model_names = model_dictionary.keys()

parser = argparse.ArgumentParser(description='Image Object Counting Model')
parser.add_argument('--model', '-m', metavar='MODEL', default='InceptionResNetV2', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: ResNet50)')
parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=4, type=int, metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lrd','--learning-rate-decay-step', default=8, type=int, metavar='N', help='learning rate decay epoch')
parser.add_argument('--momentum', default=0.1, type=float, metavar='M', help='momentum')
parser.add_argument('--max-target', default=20, type=int, metavar='N', help='smaximum number of target images for validation (default: 20)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', default=False, type=bool, metavar='BOOL', help='evaluate or train')
parser.add_argument('--pd','--prepare-data', action='store_true', help='Prepare Data for Model Evaluation')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N', help='number of data loading workers (default: 12)')

model_dictionary_modules = {m[0]:m[1] for m in inspect.getmembers(tf.keras.applications, inspect.ismodule)}

def predictTestDataSet(modelname, model, filenameList):
    datafile = open(filenameList, "r")
    content_list = datafile.read().splitlines()
    baseFilenameList = [i.split('.')[0] for i in content_list]

    testResult = []

    for filename in baseFilenameList:
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

def plotModelEvalMetric(path, trainMetricHistory, validationMetricHistory, metricName):
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
    plt.savefig(os.path.join(path, metricName +"."+"jpg"))
    plt.show(block=False)
    plt.close()


def train_validate_test_split(df, train_percent=.8, validate_percent=.1, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[:train_end]
    validate = df.iloc[train_end:validate_end]
    test = df.iloc[validate_end:]
    return train, validate, test

def mainMenu():
    global args

    args = parser.parse_args()
    print(args)

    image_df = dataVisualization()
    snapshot_path = os.path.join(INTERMEDIATE_DIR, args.model,)
    os.makedirs(snapshot_path, exist_ok=True)

    print("Image Count before Filtering ", image_df.shape[0])
    image_df = image_df[image_df['Quantity'] <= MAX_ITEM_COUNT]
    print("Image Count After Filtering based on Max Item Count", image_df.shape[0])

    train_df, val_df, test_df = train_validate_test_split(image_df)

    train_df = train_df.iloc[:MAX_TRAIN_IMAGE_COUNT]
    val_df = val_df.iloc[:MAX_VAL_IMAGE_COUNT]
    test_df = test_df.iloc[:MAX_TEST_IMAGE_COUNT]

    train_df = train_df.astype({"filename": str, "Quantity": str})
    val_df = val_df.astype({"filename": str, "Quantity": str})
    test_df = test_df.astype({"filename": str, "Quantity": str})

    train_df['filename'] = train_df['filename'].astype(str) + ".jpg"
    val_df['filename'] = val_df['filename'].astype(str) + ".jpg"
    test_df['filename'] = test_df['filename'].astype(str) + ".jpg"

    train_df['filename'].to_csv(os.path.join(snapshot_path,TRAIN_SET_FILENAME), sep=' ', index=False, header=False)
    val_df['filename'].to_csv(os.path.join(snapshot_path,VALIDATION_SET_FILENAME), sep=' ', index=False, header=False)
    test_df['filename'].to_csv(os.path.join(snapshot_path,TEST_SET_FILENAME), sep=' ', index=False, header=False)

    for i in range(len(ClassTestFileName)):
        tempDF = test_df[test_df['Quantity'] == str(i)]
        tempDF['filename'].to_csv(os.path.join(snapshot_path, TEST_SET_CLASS_BASED_FILENAME + ClassTestFileName[i] + ".txt"), sep=' ', index=False, header=False)

    print(train_df)
    print(val_df)

    # create model
    print("=> creating model '{}'".format(args.model))
    model_func = model_dictionary[args.model]

    print("Building Model")
    base_model = model_func(include_top=True,weights=None)
    base_model.summary()

    # Add a layer where input is the output of the  second last layer
    x = tf.keras.layers.Dense(MAX_ITEM_COUNT+1, activation='softmax', name='predictions')(base_model.layers[-2].output)

    # Then create the corresponding model
    my_model = Model(inputs=base_model.input, outputs=x)
    my_model.summary()

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.1,
        decay_steps=10000,
        decay_rate=1e-4)
    optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    my_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy", "mse"])

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
        datagen = ImageDataGenerator(featurewise_std_normalization=False, rescale=1. / 255, horizontal_flip=True, )

        train_generator = datagen.flow_from_dataframe(
            dataframe=train_df,
            directory="../abid_challenge/dataset/data/bin-images-resize",
            x_col="filename", y_col="Quantity", class_mode="categorical",
            target_size=(224, 224), batch_size=args.batch_size)

        val_generator = datagen.flow_from_dataframe(
            dataframe=val_df,
            directory="../abid_challenge/dataset/data/bin-images-resize",
            x_col="filename", y_col="Quantity", class_mode="categorical",
            target_size=(224, 224), batch_size=args.batch_size)


        test_generator = datagen.flow_from_dataframe(
            dataframe=test_df,
            directory="../abid_challenge/dataset/data/bin-images-resize",
            x_col="filename", y_col="Quantity", class_mode="categorical",
            target_size=(224, 224), batch_size=args.batch_size)

        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VALID = val_generator.n // val_generator.batch_size
        STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

        callbacks_list = [
            tf.keras.callbacks.CSVLogger(os.path.join(snapshot_path,"trainHistory.csv"), separator=',', append=False),
            tf.keras.callbacks.ProgbarLogger(count_mode='steps')
        ]

        hist = my_model.fit(
            x=train_generator, batch_size=train_generator.batch_size, epochs=args.epochs, verbose=1,
            callbacks=callbacks_list, validation_data=val_generator, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=args.start_epoch, steps_per_epoch=STEP_SIZE_TRAIN,
            validation_steps=STEP_SIZE_VALID, validation_batch_size=val_generator.batch_size, validation_freq=1,
            max_queue_size=12, workers=args.workers, use_multiprocessing=False
        )

        train_history = hist.history

        print('Final training loss \t', str(round((hist.history['loss'][-1]),2))+ ' %')
        print('Final Training accuracy ', str(round((hist.history['accuracy'][-1]*100),2))+ ' %')
        print('Final Validation loss \t', str(round((hist.history['val_loss'][-1]),2))+ ' %')
        print('Final Validation accuracy ', str(round((hist.history['val_accuracy'][-1]*100),2))+ '%')


        my_model.save( os.path.join(snapshot_path, args.model + "." + "h5"))
        plot_model(my_model, to_file= os.path.join(snapshot_path, args.model + "." + "png"), show_shapes=True, show_layer_names=True)

    plotModelEvalMetric(snapshot_path,train_history["loss"],train_history["val_loss"], 'loss')
    plotModelEvalMetric(snapshot_path,train_history["accuracy"],train_history["val_accuracy"], 'accuracy')
    plotModelEvalMetric(snapshot_path,train_history["mse"],train_history["val_mse"], 'mse')

    score = my_model.evaluate(x=test_generator, batch_size=args.batch_size, verbose=1, steps=STEP_SIZE_TEST,
                              workers=1, use_multiprocessing=False
    )

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    testResultDF = predictTestDataSet(args.model, my_model, os.path.join(snapshot_path,TEST_SET_FILENAME))
    testResultDF.to_csv(os.path.join(snapshot_path,"allClassResult.csv"), sep=',', index=False, header=True)

    ConfusionMatrixDisplay.from_predictions(testResultDF["ActualQuantity"], testResultDF["PredictedQuantity"])
    plt.savefig(os.path.join(snapshot_path,"AllClassResultConfMetric.jpg"))

    for i in range(len(ClassTestFileName)):
        testResultDF = predictTestDataSet(args.model, my_model,os.path.join(snapshot_path,
                                          (TEST_SET_CLASS_BASED_FILENAME + ClassTestFileName[i] + ".txt")))
        testResultDF.to_csv(os.path.join(snapshot_path,
                                         (TEST_SET_CLASS_BASED_FILENAME+ClassTestFileName[i] + "result.csv")),
                            sep=',', index=False,header=True)
        if(testResultDF.shape[0]):
            ConfusionMatrixDisplay.from_predictions(testResultDF["ActualQuantity"], testResultDF["PredictedQuantity"])
            plt.savefig(os.path.join(snapshot_path , (TEST_SET_CLASS_BASED_FILENAME+ClassTestFileName[i]+"ConfMetric.jpg")))


# Press the green button in the gutter to run the script.
import sys

import tensorflow.keras
import tensorflow as tf
import numpy as np
if __name__ == '__main__':
    print(f"Tensor Flow Version: {tf.__version__}")
    print(f"Keras Version: {tensorflow.keras.__version__}")
    print()
    print(f"Python {sys.version}")
    gpu = len(tf.config.list_physical_devices('GPU')) > 0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")

    mainMenu()

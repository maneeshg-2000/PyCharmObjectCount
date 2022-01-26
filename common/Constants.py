# Input Dataset
IMAGE_DIR = "../abid_challenge/dataset/data/bin-images-resize"
METADATA_DIR = "../abid_challenge/dataset/data/metadata"
RESIZED_IMAGE_DIR = "../ImageDataSet/bin-images_224/"
RESIZED_IMAGE_DIR_HSV = "../ImageDataSet/bin-images_224_HSV/"

# Dataset Split Filename
INTERMEDIATE_DIR = "../snapshots/"
TRAIN_SET_FILENAME = "random_trainset.txt"
VALIDATION_SET_FILENAME = "random_validationset.txt"
TEST_SET_FILENAME =  "random_testset.txt"
TEST_SET_CLASS_BASED_FILENAME = "ClassBased_testset"

MAX_VAL_IMAGE_COUNT = 5000
MAX_TEST_IMAGE_COUNT = 5000


MAX_TRAIN_IMAGE_COUNT = 50000 #"ALL"
MAX_ITEM_COUNT = 6
MAX_ITEM_TYPE_COUNT = 3
# AWS S3 Configuration Attributes
# These configuration attributes affect both uploads and downloads.
CONFIG_ATTRS = ('multipart_threshold', 'multipart_chunksize', 'max_concurrency',
                'use_threads')
# These configuration attributes affect only downloads.
DOWNLOAD_CONFIG_ATTRS = ('max_io_queue', 'io_chunksize', 'num_download_attempts')

ClassTestFileName = ["_Zero",
                     "_One",
                     "_Two",
                     "_Three",
                     "_Four",
                     "_Five",
                     "_Six"]
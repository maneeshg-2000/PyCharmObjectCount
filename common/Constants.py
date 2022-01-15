# Input Dataset
IMAGE_DIR = "../ImageDataSet//bin-images"
METADATA_DIR = "../ImageDataSet/metadata/"
RESIZED_IMAGE_DIR = "../ImageDataSet/bin-images_224/"
RESIZED_IMAGE_DIR_HSV = "../ImageDataSet/bin-images_224_HSV/"

# Dataset Split Filename
INTERMEDIATE_DIR = "../output/"
TRAIN_SET_FILENAME = INTERMEDIATE_DIR + "random_trainset.txt"
VALIDATION_SET_FILENAME = INTERMEDIATE_DIR + "random_validationset.txt"
TEST_SET_FILENAME = INTERMEDIATE_DIR + "random_testset.txt"
TEST_SET_CLASS_BASED_FILENAME = INTERMEDIATE_DIR + "ClassBased_testset"

MAX_IMAGE_SET = 25000 #"ALL"
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
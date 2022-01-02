# Input Dataset
IMAGE_DIR = "../ImageDataSet/bin-images/"
METADATA_DIR = "../ImageDataSet/metadata/"
RESIZED_IMAGE_DIR = "../ImageDataSet/bin-images_224/"
RESIZED_IMAGE_DIR_HSV = "../ImageDataSet/bin-images_224_HSV/"

# Dataset Split Filename
INTERMEDIATE_DIS = "../output/"
TRAIN_SET_FILENAME = INTERMEDIATE_DIS + "random_trainset.txt"
VALIDATION_SET_FILENAME = INTERMEDIATE_DIS + "random_validationset.txt"
TEST_SET_FILENAME = INTERMEDIATE_DIS + "random_testset.txt"

MAX_IMAGE_SET = 1000 #"ALL"
# AWS S3 Configuration Attributes
# These configuration attributes affect both uploads and downloads.
CONFIG_ATTRS = ('multipart_threshold', 'multipart_chunksize', 'max_concurrency',
                'use_threads')
# These configuration attributes affect only downloads.
DOWNLOAD_CONFIG_ATTRS = ('max_io_queue', 'io_chunksize', 'num_download_attempts')

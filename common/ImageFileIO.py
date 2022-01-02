import hashlib
import os
import platform
import shutil
import time
import json
import random
import numpy as np

from common.Constants import *

def readFileWithJson (filename):
    if not filename:
        print("Error: Incorrect Filename")
        return;

    print("File Name is \n", filename)
    metaFilename = os.path.join(METADATA_DIR, ('%s.json' % filename))
    imageFilename = os.path.join(IMAGE_DIR, ('%s.jpg' % filename))

    metadata = json.loads(open(metaFilename).read())
    quantity = metadata['EXPECTED_QUANTITY']
    print("Quanity in Image Met", quantity)

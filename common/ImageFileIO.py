import os
import json

from common.Constants import *
import json
import os

from common.Constants import *


def readFileWithJson (filename):
    if not filename:
        print("Error: Incorrect Filename")
        return;

    metaFilename = os.path.join(METADATA_DIR, ('%s.json' % filename))
    imageFilename = os.path.join(IMAGE_DIR, ('%s.jpg' % filename))

    metadata = json.loads(open(metaFilename).read())
    quantity = metadata['EXPECTED_QUANTITY']
    return [filename,quantity]

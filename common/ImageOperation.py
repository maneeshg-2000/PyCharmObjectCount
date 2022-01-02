import os
from common.Constants import *
import cv2

def resizeImages():
    imageList = os.listdir(IMAGE_DIR)
    os.makedirs(RESIZED_IMAGE_DIR, exist_ok=True)

    for filename in imageList:
        jpg_path = os.path.join(IMAGE_DIR, filename)
        resized_jpg_path = os.path.join(RESIZED_IMAGE_DIR,filename)
        img = cv2.imread(jpg_path, 1)
        resized_img = cv2.resize(img, (224,224), interpolation = cv2.INTER_NEAREST)
        cv2.imwrite(resized_jpg_path,resized_img)

def convertRGB2HSVImages():
    imageList = os.listdir(RESIZED_IMAGE_DIR)
    imageCount = len(imageList)
    os.makedirs(RESIZED_IMAGE_DIR_HSV, exist_ok=True)
    i =0
    for filename in imageList:
        i = i+1
        if i % 1000 == 0:
            print("convertRGB2HSVImages: processing (%d/%d)..." % (i, imageCount))

        jpg_path = os.path.join(RESIZED_IMAGE_DIR, filename)
        hsv_path = os.path.join(RESIZED_IMAGE_DIR_HSV,filename)
        img = cv2.imread(jpg_path, 1)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        cv2.imwrite(hsv_path,hsv_img)

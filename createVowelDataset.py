import imutils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import keras
from keras import models, layers, losses, optimizers, metrics
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.models import Sequential
from PIL import Image
import numpy as np
import os
import random
from sklearn import ensemble,preprocessing
import keras.utils.np_utils
import cv2
import pytesseract
import imutils
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
import pprint


'''Here we are building vowel dataset for Devanagari'''


def checkExistingDimensions():
    sample_img = Image.open("/Users/sunitakoppar/PycharmProjects/datasets/DevanagariHandwrittenCharacterDataset/Train/character_1_ka/1340.png")
    imarray = np.asarray(sample_img.getdata())
    #print("Image Array is ",imarray)
    pprint.pprint(imarray)
    custom_img = Image.open("./devanagari_custom_dataset/Train/character_1_ka/char_new1.png")
    custarray =   np.asarray(custom_img.getdata())
    pprint.pprint(custarray);


def createVowelDataset(vowel_label, imgpath):
    img = cv2.imread(imgpath)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
    dataset_element = np.pad(cv2.resize(thresh, (28, 28)), 2)
    #Check the displayed image
    #cv2.imshow("massaged vowel img",dataset_element);cv2.waitKey(1000)
    x=random
    iswritten = cv2.imwrite("./devanagari_custom_dataset/Train/"+vowel_label + "/134.png",dataset_element)
    print("Write is ", iswritten)

if __name__ == "__main__":
    checkExistingDimensions()
    createVowelDataset("vowel_1_a", "./HandWrittenVowels/cropped/2a.png")
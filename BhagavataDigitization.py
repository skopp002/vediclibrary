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



'''
Below code shows the contents of the file with hindi detection
'''
from PIL import Image
import pytesseract

img = Image.open("dataset/HandwrittenBhagavadGeethaShlokas.jpg")
img.load()
text = pytesseract.image_to_string(img, lang="hin")  #Specify language to look after!
# print("Lets print character at a time")
print(text)
# for c in text:
#     print("Char ",c)



src_img = cv2.imread("/Users/sunitakoppar/Documents/FPKProject/dataset/P-149-R.jpg")
#cv2.imshow('original',src_img);cv2.waitKey(0); cv2.destroyAllWindows()

copy=src_img.copy()

edged=cv2.Canny(src_img,30,200)
#cv2.imshow('canny edges',edged)
#cv2.waitKey(1); cv2.destroyAllWindows()
height = src_img[0]
width=src_img[1]


#use a copy of your image, e.g. - edged.copy(), since finding contours alter the image
#we have to add _, before the contours as an empty argument due to upgrade of the open cv version_, contours,hierarchy=cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
#cv2.imshow('canny edges after contouring', edged)
#cv2.waitKey(0); cv2.destroyAllWindows()


gray_img = cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)

#cv2.imshow('gray image',gray_img);cv2.waitKey(0); cv2.destroyAllWindows()

#binary
ret,thresh = cv2.threshold(src_img,127,255,cv2.THRESH_BINARY_INV)
#cv2.imshow('second',thresh) ;cv2.waitKey(0)
# print("Resizing the image")
# src_img = cv2.resize(copy,dsize=(1320,int(1320*height/width)),interpolation=cv2.INTER_AREA)

boundedimg = cv2.boundingRect(gray_img)
cv2.imshow("boundedimg",boundedimg)

#dilation
kernel = np.ones((5,100), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
#cv2.imshow('dilated',img_dilation) ;cv2.waitKey(0)

SZ = 20 # Character size. I am not sure if this is right number. Need to test
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) <  1e-2:
        # no deskewing needed.
        return img.copy()
    # Calculate skew based on central momemts.
    skew = m['mu11']/m['mu02']
    print("Skew is ",skew)
    # Calculate affine transform to correct skewness.
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

deskew(src_img)
#find contours
im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = src_img[y:y+h, x:x+w]

    # show ROI
    cv2.imshow('segment no:'+str(i),roi)
    cv2.rectangle(src_img,(x,y),( x + w, y + h ),(90,0,255),2)
    cv2.waitKey(0)

cv2.imshow('marked areas',src_img)
cv2.waitKey(0)

height = src_img[0]
width=src_img[1]


print("Applying Adaptive Threshold with Kernal :- 21 X 21")
bin_img=cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, 21,20)
bin_img1 = bin_img.copy()
bin_img2 = bin_img.copy()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
kernel1 = np.array([[1,0,1],[0,1,0],[1,0,1]],dtype=np.uint8)

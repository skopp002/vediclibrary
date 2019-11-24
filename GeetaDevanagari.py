'''
 The images we use here have multiple lines with multiple words in each line.
 Each word has multiple characters and every character is comprised of multiple pixels.

 OpenCV is the opensource computer vision library which helps represent any image into matrix
 The approach here is, every character is comprised of multiple pixels and each pixel has 3 dimensions each
 representing a color, red, green and blue. Thus every pixel becomes a vector with 3 dimensions.
 OpenCV provides a set of useful APIs to work with the matrices like binarization, dilation, grayscaling etc which helps in character recognition
 '''
from PIL import Image
import pytesseract
'''
pytesseract just provides python bindings to the extent of calling tesseract installed locally on the system.
tessearct itself is an open source library in C/C++
'''

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


'''
Tesseract is an opensource OCR Engine with out of box capabilities to recognize upto 100 languages.
pytesseract is the python binding to the tesseract installed on the operating system.
Lets try using it out of the box and understand the accuracy. 
We will first print the entire document converted and printed as is

The out of box capabilities seem very limiting for Devanagari. To confirm its a language challenge, 
lets try recognizing the same shlokas written in english
'''

def attemptOutOfBoxOCR():
    #geeta_img = cv2.imread("dataset/Devanagari/HandwrittenGeetaShlokas_1.jpg")
    simple_sanskrit =  cv2.imread("dataset/Devanagari/simpledevanagari.jpg")
    simple_sanskrit = imutils.resize(simple_sanskrit,height = 500)
    #cv2.imshow("document",geeta_img);cv2.waitKey(500) ; cv2.destroyAllWindows()
    # height = geeta_img[0]
    # width=geeta_img[1]
    # text = pytesseract.image_to_string(geeta_img, lang="hin")  #Specify language to look after!
    # print("Lets print character at a time")
    #print(text)
    simple_text = pytesseract.image_to_string(simple_sanskrit, lang="san")
    print(simple_text)
    return simple_sanskrit
    '''OCR Output for this out of the box:
    Using hin as language
    कमल नलयज कमर मढात   
    कमल रमणा नाशगयाए

    Using san as language
    कमृ चयन कलर नदा
    नरम रमा नार) ला
    '''


def wordSegmentation(simple_sanskrit):
    gray_img = cv2.cvtColor(simple_sanskrit,cv2.COLOR_BGR2GRAY)
    #binarize
    ret, thresh = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY_INV)
    print("Thresholded version")
    print(pytesseract.image_to_string(thresh, lang="san"))
    blurred_gray = cv2.GaussianBlur(gray_img, (5, 5), 0)
    print("Blurred version")
    print(pytesseract.image_to_string(blurred_gray, lang="san"))
    edged = cv2.Canny(blurred_gray, 75, 200)
    print("Edged version")
    print(pytesseract.image_to_string(edged, lang="san"))
    #dilation
    kernel = np.ones(thresh.shape[:1], np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    #find contours
    #ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ctrs = imutils.grab_contours(cnts)
    #sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr))
    print("Num of sorted countors are ",len(sorted_ctrs))
    lines={}
    for i, ctr in enumerate(sorted_ctrs):
        #draw_contour(thresh, ctr, i)
        #Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        # Getting ROI for words only and not smaller characters
        if(w >= 100):
            roi = thresh[y:y+h, x:x+w]
            lines[i]=cv2.resize(roi,(50,50))
            # show Region of the image that has been extracted
            cv2.imshow('segment no:'+str(i),roi)
            cv2.waitKey(1000)
            write = cv2.imwrite("./selected_contour_segments/segment_no_"+str(i)+".png",roi)
            #print("Write status ", write)
            cv2.rectangle(thresh,(x,y),( x + w, y + h ),(90,0,255),2)


def characterSegmentation():
    path = "./selected_contour_segments/"
    wordsegments = os.listdir(path)
    for wordimg in wordsegments:
        newPath = path + wordimg  #Create new path by adding folder name
        #Read the image in gray scale
        im = cv2.imread(newPath, 0)
        #Below 2 lines show the words read in
        #cv2.imshow(wordimg,im)
        #cv2.waitKey(1000)
        #cv2.calcHist(images, channels, mask, bins, ranges)
        hist = cv2.calcHist([im], [0], None, [256], [0, 256])
        # counts, bins = np.histogram(im)
        # plt.hist(bins[:-1], bins, weights=counts)
        plt.hist(im.ravel(), 256, [0, 256]);
        plt.show()
        out = sum(im, 2)



if __name__ == "__main__":
    print ("Starting the Handwritten Text Classifier")
    #handwritten_img = attemptOutOfBoxOCR()
    #wordSegmentation(handwritten_img)
    characterSegmentation()



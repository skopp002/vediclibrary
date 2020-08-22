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
import codecs
import io
import pprint
import csv
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
import numpy as np
import os
import random
from sklearn import ensemble,preprocessing
import keras.utils.np_utils
import cv2
import pytesseract
import imutils
from PIL import Image, ImageFont, ImageDraw
from keras.preprocessing import image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time
from keras.models import Model
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
'''
Tesseract is an opensource OCR Engine with out of box capabilities to recognize upto 100 languages.
pytesseract is the python binding to the tesseract installed on the operating system.
Lets try using it out of the box and understand the accuracy.
We will first print the entire document converted and printed as is

The out of box capabilities seem very limiting for Devanagari. To confirm its a language challenge,
lets try recognizing the same shlokas written in english
'''

#Global Variables
X_train=[]
X_test=[]
T_train=[]
T_test=[]
cnn=Sequential()
knn=KNeighborsClassifier(n_neighbors=36)
random=RandomForestClassifier(n_estimators=150, random_state=0) #max_depth=10,
le=preprocessing.LabelEncoder()
le_name_mapping=dict()

with codecs.open('sourcetexts/UnicodeDevanagariSymbolsMap.csv', 'r',encoding='utf-8') as f:  # newline='' to be added foer io.open
    unicodemap = dict(csv.reader(f))

def attemptOutOfBoxOCR():
    #geeta_img = cv2.imread("dataset/Devanagari/HandwrittenGeetaShlokas_1.jpg")
    simple_sanskrit =  cv2.imread("sourcetexts/Devanagari/simpledevanagari.jpg")
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

#Code referenced from https://github.com/dishank-b/Character_Segmentation/blob/master/main.py
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
            # cv2.imshow('segment no:'+str(i),roi)
            # cv2.waitKey(1000)
            write = cv2.imwrite("./selected_contour_segments/segment_no_"+str(i)+".png",roi)
            #print("Write status ", write)
            cv2.rectangle(thresh,(x,y),( x + w, y + h ),(90,0,255),2)


def characterSegmentation(classifier):
    path = "./selected_contour_segments/"
    wordsegments = os.listdir(path)
    for i, wordimg in enumerate(wordsegments):
        newPath = path + wordimg  #Create new path by adding folder name
        #Read the image in gray scale
        im = cv2.imread(newPath, 0)
        height, width = im.shape[:2]
        startRow = int(height * .15)
        startCol = int(width * 0.0001)
        endRow = int(height)
        endCol = int(width)
        croppedImage = im[startRow:endRow, startCol:endCol]
        #Below 2 lines show the words read in
        contours = cv2.findContours(croppedImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ctrs = imutils.grab_contours(contours)
        word_dir=  "./segmented_chars/word_"+ str(i) + "/"
        print ("Displaying word ", i)
        if not os.path.exists(word_dir):
            os.makedirs(word_dir)
        # sort contours
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr))
        for j, cnt in enumerate(sorted_ctrs):
            if (cv2.contourArea(cnt) > 35):
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(croppedImage, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = croppedImage[y:y + h, x:x + w]
                if(w <= 180):
                   img = cv2.resize(roi, (224, 224))
                   write_status = cv2.imwrite(word_dir + "/char_new" + str(j) + ".png", img)
                   predictCharacters(classifier, img, i, j)


def predictCharacters(clsfr, char_img, word_count, char_count):
    imgData = np.array(char_img)
    flat = imgData.flatten()
    np.shape(flat)
    label = ''
    if (clsfr == 'googlenet'):
        flat = char_img.reshape(1, -1)  # (1, -1)
        random_pred_mat = random.predict(flat)
        random_pred = np.argmax(random_pred_mat[0], axis=-1)
        label = le_name_mapping.get(random_pred, str(random_pred))
        # https://unicode.org/charts/PDF/U0900.pdf
        print(unicodemap[label])
# cv2.imshow("displaying " + label + str(word_count) + str(char_count), char_img);
    #cv2.waitKey(1000)


def prepareClassifiers(classifier):
   if(classifier == "knn"):
        knnClassifier() #--> Takes too long on full dataset with n not specified
   elif (classifier == "random"):
       randomForestClassifier()
   elif (classifier == "googlenet"):
       googleNetClassifier()

def genericClassifier(clfr,  acc_str, matrix_header_str):
    """run chosen classifier and display results"""
    start_time = time.time()
    clfr.fit(X_train, T_train)
    y_pred = clfr.predict(X_test)
    print("Time taken for prediction = %f seconds" % (time.time() - start_time))
    print(acc_str.format(accuracy_score(T_test, y_pred) * 100))
    acc=accuracy_score(T_test, y_pred) * 100
    print("acc=",acc)
    return y_pred,acc

#Code reference https://github.com/PriSawant7/ML-Devanagari-Character-Recognition/blob/master/Devanagari_Character_Recoginition.ipynb
def prepareTestTrainData(classifier):
    path = "./devanagari_custom_dataset/Train/"
    folders = os.listdir(path)
    imageList = []
    imageMatrix = []
    newIm = []
    labels = []
    # Get list of folders in current path
    from keras.preprocessing import image
    for folder in folders:
        if folder.startswith("character"):
            newPath = path + folder  # Create new path by adding folder name
            folderName = os.path.split(os.path.abspath(newPath))[1]
            print("Folder is ",folderName )
            characterName = folderName.split("_")[2]
            imageList = [f for f in os.listdir(newPath) if
                     os.path.splitext(f)[-1] == '.png']  # Check if PNG files only then add
            for img in imageList:  # Traverse the list of files and add each file name to the imageFile
               #The imagenet based algorithms have been trained on RGB images of 224 by 224.
               #For transfer learning using the algorithms like Inception or VGG, we will have to
               #reshape the original images into the same dimensions.
                im = image.load_img(newPath + "//" + img,target_size=(224,224))
                labels.append(characterName)
                imageArray = np.asarray(im.getdata())
                flattenedImageArray = imageArray.flatten()
                imageMatrix.append(flattenedImageArray)
    print("Size of the image matrix = ", np.size(imageMatrix))
    a = np.array(imageMatrix)
    n_samples = len(a)
    X = a.reshape((n_samples, -1))
    T = np.array(labels)
    print('Features size = ', X.shape)
    print('Labels size = ', T.shape)
    print(T)
    global le
    le.fit(T)
    T = le.transform(T)
    T = keras.utils.np_utils.to_categorical(T)
    global le_name_mapping
    le_name_mapping = dict(zip(le.transform(le.classes_),le.classes_))
    print("Labels are" ,le_name_mapping)
    global X_train, X_test, T_train, T_test
    X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.3, random_state=34)
    prepareClassifiers(classifier)


def knnClassifier():
  print('KNN Classifier starting ...')
  global knn
  y_pred, knnacc = genericClassifier(knn, "CNN-KNN Accuracy: {0:0.1f}%", "SVM Confusion matrix")

def randomForestClassifier():
  print('Random Forest Classifier starting ...')
  global random
  y_pred, randomacc = genericClassifier(random, "CNN-Random Accuracy: {0:0.1f}%", "SVM Confusion matrix")


def googleNetClassifier():
    from keras.applications.inception_v3 import InceptionV3
    # load model
    gnet = InceptionV3(input_shape=[224,224]+[3], weights='imagenet', include_top=False)
    #Setting this to false ensures the model is not getting trained since we want to leverage pre-trained model
    for layer in gnet.layers:
        layer.trainable = False
    global T_train
    global X_train
    prediction = Dense(len(T_train), activation='softmax')(X_train)
    # summarize the model
    model = Model(inputs=gnet.input, outputs=prediction)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()


#TODO Use imrotate and max vertical projection by picking the theta at which the vertical projection is highest
import unicodedata
if __name__ == "__main__":
    classifier="googlenet" #"random"#"cnn" #
    print ("Starting the Handwritten Text Classifier")
    handwritten_img = attemptOutOfBoxOCR()
    wordSegmentation(handwritten_img)
    #lets prepare the classifiers
    prepareTestTrainData(classifier)
    characterSegmentation(classifier)



###References
#https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/
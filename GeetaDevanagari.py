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


def characterSegmentation():
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
        if not os.path.exists(word_dir):
            os.makedirs(word_dir)
        # sort contours
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr))
        for i, cnt in enumerate(sorted_ctrs):
            if (cv2.contourArea(cnt) > 40):
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(croppedImage, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = croppedImage[y:y + h, x:x + w]
                # cv2.imshow('character no:' + str(i), roi)
                # cv2.waitKey(1000)
                if(w <= 200):
                    cv2.imshow(word_dir+"/char"+str(i)+".png", roi)
                    print("char image shape is ", roi.shape)
                   # img = roi.resize((32, 32))
                   # img = roi.convert('L')
                    imgData = np.array(roi)
                    plt.imshow(imgData)
                    # inverted_image.save('new1.jpg')
                    # np.set_printoptions(threshold=100)
                    flat = imgData.flatten()
                    np.shape(flat)
                    flat = flat.reshape(1, -1)
                    buildclassifier("knn",flat)




def genericClassifier(clfr, x_train_data, y_train_data, x_test_data, y_test_data, acc_str, matrix_header_str):
    """run chosen classifier and display results"""
    start_time = time.time()
    clfr.fit(x_train_data, y_train_data)
    y_pred = clfr.predict(x_test_data)
    print("Time taken for prediction = %f seconds" % (time.time() - start_time))
    print(acc_str.format(accuracy_score(y_test_data, y_pred) * 100))
    acc=accuracy_score(y_test_data, y_pred) * 100
    print("acc=",acc)
    return y_pred,acc

#Code reference https://github.com/PriSawant7/ML-Devanagari-Character-Recognition/blob/master/Devanagari_Character_Recoginition.ipynb
def buildclassifier(classifier_name, char_img):
    path = "/Users/sunitakoppar/PycharmProjects/datasets/DevanagariHandwrittenCharacterDataset/Train/"
    folders = os.listdir(path)
    imageList = []
    imageMatrix = []
    newIm = []
    labels = []
    # Get list of folders in current path
    for folder in folders:
        if folder.startswith("character"):
            newPath = path + folder  # Create new path by adding folder name
            folderName = os.path.split(os.path.abspath(newPath))[1]
            print("Folder is ",folderName )
            characterName = folderName.split("_")[2]
            imageList = [f for f in os.listdir(newPath) if
                     os.path.splitext(f)[-1] == '.png']  # Check if PNG files only then add
            for image in imageList:  # Traverse the list of files and add each file name to the imageFile
                im = Image.open(newPath + "//" + image)
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
    le = preprocessing.LabelEncoder()
    le.fit(T)
    T = le.transform(T)
    T = keras.utils.np_utils.to_categorical(T)
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)
    X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.3, random_state=34)
    img_height_rows = 32
    img_width_cols = 32
    im_shape = (img_height_rows, img_width_cols, 1)
    print(im_shape)
    x_train = X_train.reshape(X_train.shape[0], *im_shape)  # Python TIP :the * operator unpacks the tuple
    x_test = X_test.reshape(X_test.shape[0], *im_shape)
    if(classifier_name == "cnn"):
        cnn = Sequential()
        kernelSize = (3, 3)
        ip_activation = 'relu'
        ip_conv_0 = Conv2D(filters=4, kernel_size=kernelSize, input_shape=im_shape, activation=ip_activation)
        cnn.add(ip_conv_0)
        # Add the next Convolutional+Activation layer
        ip_conv_0_1 = Conv2D(filters=4, kernel_size=kernelSize, activation=ip_activation)
        cnn.add(ip_conv_0_1)
        # Add the Pooling layer
        pool_0 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
        cnn.add(pool_0)
        ip_conv_1 = Conv2D(filters=4, kernel_size=kernelSize, activation=ip_activation)
        cnn.add(ip_conv_1)
        ip_conv_1_1 = Conv2D(filters=4, kernel_size=kernelSize, activation=ip_activation)
        cnn.add(ip_conv_1_1)
        pool_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
        cnn.add(pool_1)
        flat_layer_0 = Flatten()
        cnn.add(Flatten())
        # Now add the Dense layers
        h_dense_0 = Dense(units=20, activation=ip_activation, kernel_initializer='uniform')
        cnn.add(h_dense_0)
        # Let's add one more before proceeding to the output layer
        h_dense_1 = Dense(units=1024, activation=ip_activation, kernel_initializer='uniform', name='dense11')
        cnn.add(h_dense_1)
        n_classes = 36
        op_activation = 'softmax'
        output_layer = Dense(units=n_classes, activation=op_activation, kernel_initializer='uniform')
        cnn.add(output_layer)
        opt = optimizers.Adagrad(lr=0.001)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
        # Compile the classifier using the configuration we want
        cnn.compile(optimizer=opt, loss=loss, metrics=metrics)
        history = cnn.fit(x_train, T_train,
                      batch_size=36, epochs=36,
                      validation_data=(x_test, T_test))
        scores = cnn.evaluate(x_test, T_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))
        y_pred = cnn.predict(char_img)
        print("Prediction is ",y_pred[0])
    # model = cnn
    # layer_name = 'dense11'
    # intermediate_layer_model = Model(inputs=cnn.input,
    #                                  outputs=model.get_layer(layer_name).output)
    # intermediate_output = intermediate_layer_model.predict(x_train)
    # elif(classifier_name == "random_forest"):
    #     print('KNN Classifier starting ...')
    #     partialDataKNNClassifier = KNeighborsClassifier()
    #     y_pred, knnacc = genericClassifier(partialDataKNNClassifier, X_train, T_train, X_test, T_test,
    #                                    "CNN-KNN Accuracy: {0:0.1f}%", "SVM Confusion matrix")
    #     y_pred = cnn.predict(char_img)
    #     print("Prediction is ", y_pred[0])
    elif (classifier_name == "knn"):
        print('KNN Classifier starting ...')
        partialDataKNNClassifier = KNeighborsClassifier()
        y_pred, knnacc = genericClassifier(partialDataKNNClassifier, X_train, T_train, X_test, T_test,
                                           "CNN-KNN Accuracy: {0:0.1f}%", "SVM Confusion matrix")
        le.inverse_transform(y_pred[0])
        char_pred = partialDataKNNClassifier.predict(char_img)
        print("knn prediction for given character ", char_pred)




if __name__ == "__main__":
    print ("Starting the Handwritten Text Classifier")
    handwritten_img = attemptOutOfBoxOCR()
    wordSegmentation(handwritten_img)
    characterSegmentation()







 # # sort contours
 #        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr))
 #        print("Num of sorted countors are ", len(sorted_ctrs))
 #        lines = {}
 #        for i, ctr in enumerate(sorted_ctrs):
 #            x, y, w, h = cv2.boundingRect(ctr)
 #            # Getting ROI for words only and not smaller characters
 #            if (w >= 70 & h >=70 & w < 100 & h <100):
 #                roi = im[y:y + h, x:x + w]
 #                lines[i] = cv2.resize(roi, (50, 50))
 #                # show Region of the image that has been extracted
 #                cv2.imshow('segment no:' + str(i), roi)
 #                cv2.waitKey(1000)
 #                write = cv2.imwrite("./character_contour_segments/segment_no_" + str(i) + ".png", roi)
 #                # print("Write status ", write)
 #                cv2.rectangle(im, (x, y), (x + w, y + h), (90, 0, 255), 2)



#This did not eliminate shirorekha
# ret,thresh_img = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
#         morph_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
#         cv2.imshow('morphed image' ,morph_img)
#         cv2.waitKey(1000)
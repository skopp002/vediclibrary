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

geeta_img = cv2.imread("dataset/Devanagari/HandwrittenGeetaShlokas_1.jpg")
simple_sanskrit =  cv2.imread("dataset/Devanagari/simpledevanagari.jpg")
#cv2.imshow("document",geeta_img);cv2.waitKey(500) ; cv2.destroyAllWindows()
height = geeta_img[0]
width=geeta_img[1]
text = pytesseract.image_to_string(geeta_img, lang="hin")  #Specify language to look after!
# print("Lets print character at a time")
#print(text)
simple_text = pytesseract.image_to_string(simple_sanskrit, lang="hin")
print(simple_text)

#Sanskrit language in english script
eng_img1 = cv2.imread("dataset/English/GeethaEnglishHandwritten_1.jpg")
test_eng = pytesseract.image_to_string(eng_img1, lang="eng")
#print(test_eng)

#English script and language
eng_img2 =  cv2.imread("dataset/English/testocr_1.jpg")
test_ocr = pytesseract.image_to_string(eng_img2, lang="eng")
#print(test_ocr)


'''
The performance out of box is lot better with english as script as well as the language
Next we will split the characters individually and use the UCI dataset to train the models
The below segment has been sourced from 
https://github.com/watersink/Character-Segmentation/blob/master/test_char_seg.py
and has been tweaked to suit the needs of the project
'''

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Char_segment(object):
    def __init__(self):
        self.input_shape = (2048, 64, 3)
        self.batch_size = 1
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.model = self.network()
            init = tf.global_variables_initializer()
            self.session = tf.Session(graph=self.graph)
            self.session.run(init)
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(save_path='./save/char_seg.ckpt-4000', sess=self.session)

    def network(self):
        network = {}
       # network["inputs"] = tf.placeholder(tf.float32, [self.batch_size, self.input_shape[1], self.input_shape[0],
        #                                                self.input_shape[2]], name='inputs')
        network["down-conv1"] = tf.layers.conv2d(inputs=network["inputs"], filters=32, kernel_size=(2, 2),
                                                 padding="same", activation=tf.nn.relu, name="down-conv1")
        network["down-pool1"] = tf.layers.max_pooling2d(inputs=network["down-conv1"], pool_size=[2, 2], strides=2)
        network["down-conv2"] = tf.layers.conv2d(inputs=network["down-pool1"], filters=64, kernel_size=(2, 2),
                                                 padding="same", activation=tf.nn.relu, name="down-conv2")
        network["down-pool2"] = tf.layers.max_pooling2d(inputs=network["down-conv2"], pool_size=[2, 2], strides=2)
        network["down-conv3"] = tf.layers.conv2d(inputs=network["down-pool2"], filters=128, kernel_size=(2, 2),
                                                 padding="same", activation=tf.nn.relu, name="down-conv3")
        network["down-pool3"] = tf.layers.max_pooling2d(inputs=network["down-conv3"], pool_size=[2, 2], strides=2)
        network["down-conv4"] = tf.layers.conv2d(inputs=network["down-pool3"], filters=256, kernel_size=(2, 2),
                                                 padding="same", activation=tf.nn.relu, name="down-conv4")
        network["down-pool4"] = tf.layers.max_pooling2d(inputs=network["down-conv4"], pool_size=[2, 2], strides=2)
        network["down-conv5"] = tf.layers.conv2d(inputs=network["down-pool4"], filters=512, kernel_size=(2, 2),
                                                 padding="same", activation=tf.nn.relu, name="down-conv5")
        network["down-pool5"] = tf.layers.max_pooling2d(inputs=network["down-conv5"], pool_size=[2, 2], strides=2)
        network["down-conv6"] = tf.layers.conv2d(inputs=network["down-pool5"], filters=512, kernel_size=(2, 2),
                                                 padding="same", activation=tf.nn.relu, name="down-conv6")
        network["down-pool6"] = tf.layers.max_pooling2d(inputs=network["down-conv6"], pool_size=[2, 2], strides=2)

        network["up-conv1"] = tf.layers.conv2d_transpose(inputs=network["down-pool6"], filters=512, kernel_size=(1, 2),
                                                         strides=(1, 2), padding="valid", activation=tf.nn.relu,
                                                         name="up-conv1")
        network["up-conv2"] = tf.layers.conv2d_transpose(inputs=network["up-conv1"], filters=512, kernel_size=(1, 2),
                                                         strides=(1, 2), padding="valid", activation=tf.nn.relu,
                                                         name="up-conv2")
        network["up-conv3"] = tf.layers.conv2d_transpose(inputs=network["up-conv2"], filters=256, kernel_size=(1, 2),
                                                         strides=(1, 2), padding="valid", activation=tf.nn.relu,
                                                         name="up-conv3")
        network["up-conv4"] = tf.layers.conv2d_transpose(inputs=network["up-conv3"], filters=128, kernel_size=(1, 2),
                                                         strides=(1, 2), padding="valid", activation=tf.nn.relu,
                                                         name="up-conv4")
        network["up-conv5"] = tf.layers.conv2d_transpose(inputs=network["up-conv4"], filters=64, kernel_size=(1, 2),
                                                         strides=(1, 2), padding="valid", activation=tf.nn.relu,
                                                         name="up-conv5")
        network["up-conv6"] = tf.layers.conv2d_transpose(inputs=network["up-conv5"], filters=1, kernel_size=(1, 2),
                                                         strides=(1, 2), padding="valid", activation=None,
                                                         name="up-conv6")

        network["outputs"] = tf.nn.sigmoid(tf.contrib.layers.flatten(network["up-conv6"]))
        return network

    def recognize_image(self, img):
        # img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]), interpolation=cv2.INTER_AREA)
        image = resized_img.reshape(1, self.input_shape[1], self.input_shape[0], self.input_shape[2])
        image = image.astype('float32') / 255.0
        with self.graph.as_default():
            feed = {self.model["inputs"]: image}
            mat_results = self.session.run(self.model["outputs"], feed_dict=feed)

        print(np.max(np.max(mat_results)), np.min(np.min(mat_results)))
        mat_results[mat_results >= 0.5] = 1
        mat_results[mat_results < 0.5] = 0
        boxes = self.generate_box(mat_results)
        return boxes, mat_results[0] * 255

    def generate_box(self, mat_results):
        boxes = []
        box = [0, 0, 0, self.input_shape[1]]

        mat_results = mat_results[0]

        for i in range(len(mat_results) - 1):
            if box[0] != 0 and box[2] != 0:
                boxes.append(box)
                box = [0, 0, 0, self.input_shape[1]]

            if mat_results[i] == 0:
                continue
            elif mat_results[i] == 1 and box[0] == 0:
                box[0] = i
            elif mat_results[i] == 1 and box[0] != 0 and mat_results[i + 1] == 0:
                box[2] = i + 1
            else:
                continue

        return boxes

if __name__ == '__main__':
    cs = Char_segment()
    img = geeta_img
    img = cv2.resize(img, (2048, 48), interpolation=cv2.INTER_AREA)
    assert (img is not None)
    boxes, mat_results = cs.recognize_image(img)
    for i in range(len(mat_results)):
        if mat_results[i] > 128:
            cv2.circle(img, (i, 24), 1, (0, 0, 255), -1)

    for box in boxes:
        print(box)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    cv2.imwrite("4.png", img)
    cv2.imshow("result", img)
    cv2.waitKey()
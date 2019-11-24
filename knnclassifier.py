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

path = "/Users/sunitakoppar/PycharmProjects/datasets/DevanagariHandwrittenCharacterDataset/Train/"
folders = os.listdir(path)
imageList=[]
imageMatrix = []
newIm = []
labels=[]
# Get list of folders in current path
for folder in folders:
    newPath = path + folder  #Create new path by adding folder name
    folderName = os.path.split(os.path.abspath(newPath))[1]
    characterName = folderName #.split("_")[2]
    imageList=[f for f in os.listdir(newPath) if os.path.splitext(f)[-1] == '.png'] #Check if PNG files only then add
    for image in imageList:   #Traverse the list of files and add each file name to the imageFile
        im = Image.open(newPath+"//"+image)
        labels.append(characterName)
        imageArray = np.asarray(im.getdata())
        flattenedImageArray = imageArray.flatten()
        imageMatrix.append(flattenedImageArray)

print("Size of the image matrix = ",np.size(imageMatrix))
a = np.array(imageMatrix)
print (a)
print(a.shape)
#define variables
n_samples = len(a)
X = a.reshape((n_samples,-1))
T = np.array(labels)
print('Features size = ',X.shape)
print('Labels size = ',T.shape)
print(T)
le=preprocessing.LabelEncoder()
le.fit(T)
T=le.transform(T)

T=keras.utils.np_utils.to_categorical(T)

le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)


#from sklearn.cross_validation import train_test_split
X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.3, random_state=34)

print('Train size = ',X_train.shape," ",T_train.shape)
img_height_rows = 32
img_width_cols = 32
im_shape = (img_height_rows, img_width_cols, 1)
print(im_shape)
x_train = X_train.reshape(X_train.shape[0], *im_shape) # Python TIP :the * operator unpacks the tuple
x_test = X_test.reshape(X_test.shape[0], *im_shape)
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
h_dense_1 = Dense(units=1024, activation=ip_activation, kernel_initializer='uniform',name='dense11')
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
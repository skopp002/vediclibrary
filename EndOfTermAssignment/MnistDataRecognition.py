#Cross Referenced from my other assignment https://github.com/skopp002/appliedml/blob/master/homeworks/week4/UnsupervisedKnn.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import struct
import matplotlib as mpl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from scipy.linalg import  eigh
import seaborn as sn
from sklearn import metrics

class Digit:
    def __init__(self):
        self.label = ''
        self.data = []

    def distance(self, digit):
        sum = 0
        for i in range(0, len(self.data)):
            sum = sum + (self.data[i] - digit.data[i]) ** 2
        return sum ** .5

# minkowski order 2
def euclidean_distance(X, Y):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(X, Y)))

# minkowski order n
def minkowski_distance(X, Y, order=3):
        return sum((x - y) ** order for x, y in zip(X, Y)) ** 1 / order

# minkowski order 1
def manhattan_distance(X, Y):
        return sum(abs(x - y) for x, y in zip(X, Y))

def readIdxFiles(filename):
    with open(filename,'rb')as f:
        zero, data, dims = struct.unpack('>HBB',f.read(4))
        shape = tuple(struct.unpack('>I',f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(),dtype=np.uint8).reshape(shape)

def loadMnistFromFiles():
    raw_train = readIdxFiles("/Users/sunitakoppar/PycharmProjects/datasets/train-images-idx3-ubyte")
    #here we are flattening every 28 x 28 2-dimensional array into 1 x (28*28) 1 dimension array
    X_train = np.reshape(raw_train, (60000,28*28))
    Y_train = readIdxFiles("/Users/sunitakoppar/PycharmProjects/datasets/train-labels-idx1-ubyte")
    raw_test = readIdxFiles("/Users/sunitakoppar/PycharmProjects/datasets/t10k-images-idx3-ubyte")
    #test data has 10000 records
    X_test = np.reshape(raw_test,(10000,28*28))
    Y_test = readIdxFiles("/Users/sunitakoppar/PycharmProjects/datasets/t10k-labels-idx1-ubyte")
    return X_train,X_test,Y_train,Y_test


def loadMnistFromOpenml():
    try:
        #sklearn provides some open datasets which can be pulled based on dataset id
        mnist = fetch_openml('mnist_784', version=1, cache=True)
        mnist.target = mnist.target.astype(np.int8)
    except ImportError:
        from sklearn.datasets import fetch_mldata
        mnist = fetch_mldata('MNIST original')
        #The loading of the file was taking too long. Hence trimmed the dataset to use first 5000 records only
    return  train_test_split(mnist["data"][:15000], mnist["target"][:15000], test_size=0.2)


'''MNIST data have a bunch of handwritten digit images of size 28 * 28 = 784 in grayscale. 
Which is every pixel in the images. If each pixel represents a categorical dimension since 
it can a 0 or 1, we would end up with 784 dimensions. Hence lets use PCA, visualize it and 
determine the significant components or pixels

The data that has been provided for the assignment is a csv with 42000 datapoints flattened into
a column vector. And each column is a dimension.

'''


if __name__ == '__main__':
    d0 = pd.read_csv('./mnist_data/train.csv')
    print (d0.head(2))
    label = d0['label']
    #Separate the label for from the data
    data=d0.drop("label", axis=1)
    #This method standardizes each dimension independently. Which
    # is, all rows are normalized to be 0 on mean.
    std_data = StandardScaler().fit_transform(data)
    #To get covariance lets do std_data * transpose of std_data
    cov_data = np.matmul(std_data.T, std_data)
    print("Covariance matrix dimensions", cov_data.shape)
    #Finding eigen vectors and their values tells us independently
    #or individually impacting dimension for a variable.
    #Since eigh provides values in ascending order, we are picking last 2
    top_values, top_vectors = eigh(cov_data,eigvals=(782,783))
    print("Eigne vectors ", top_vectors.shape)

    #Lets get the projection of this data onto 2 dimensions
    top_vectors=top_vectors.T
    main_comp_data = np.matmul(top_vectors,std_data.T)
    main_comp_data = np.vstack((main_comp_data,label)).T
    print("Shape of new component ",main_comp_data.shape)
    prin_data = pd.DataFrame(data=main_comp_data, columns=("p1","p2","label"))
    sn.FacetGrid(prin_data,hue="label", size=6).map(plt.scatter, "p1","p2").add_legend()
    plt.show()

    ###Using PCA to do it out of the box. Lets use 3 components this time

    pca = PCA()
    pca.n_components = 2
    pca_data = pca.fit_transform(std_data)
    print("PCA data ", pca_data.shape)
    pca_data = np.vstack((pca_data.T,label)).T
    pca_df = pd.DataFrame(data=pca_data, columns=("p1","p2","label"))
    sn.FacetGrid(pca_df, hue="label", size=6).map(plt.scatter, "p1", "p2").add_legend()
    plt.show()


    ##Lets Use Random Forest to predict. Standardizing the data did not impact
    #Random Forest Analysis
    X_train, X_test, y_train, y_test = train_test_split(std_data, label)
    #X_train, X_test, y_train, y_test = train_test_split(data, label)
    rndClf = RandomForestClassifier()  # n_jobs=2, random_state=0
    rndClf.fit(X_train, y_train)
    rnd_pred = rndClf.predict(X_test)
    print("Confusion matrix with Random Forest")
    print(confusion_matrix(y_test, rnd_pred))
    print(classification_report(y_test, rnd_pred))
    print("Accuracy:",metrics.accuracy_score(y_test, rnd_pred))
    test_df = pd.read_csv("./mnist_data/test.csv")
    print("Test data looks like this ",test_df.head(2))
    print("Shape of the test is ", test_df.shape)
    img_id_df = pd.read_csv("./mnist_data/sample_submission.csv")
    print("Image df looks like ", img_id_df.shape)
    mod_img_id_df=img_id_df.drop("Label", axis=1)
    pred_test = rndClf.predict(test_df)
    print("sample result predictions", pred_test.shape, "and modified image_id_df ", mod_img_id_df.shape)
    sample_submission_df = np.vstack((mod_img_id_df.T,pred_test)).T
    final_submission = pd.DataFrame(data=sample_submission_df , columns=("ImageId", "Predicted_Label"))
    final_submission.to_csv('./mnist_data/final_submission.csv', header=True, index=False)







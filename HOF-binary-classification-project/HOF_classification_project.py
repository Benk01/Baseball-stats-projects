from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import utils
from sklearn import svm
from sklearn.linear_model import Perceptron

import numpy as np
import pandas as pd

# imports data and removes 1st row of feature names
data = pd.read_csv("data/projectdataset.csv", names=['Player',  'PA',  'dWAR',  'R', 'H', '1B', '2B', '3B', 'HR', 'RBI', 'SB', 'BB', 'BA', 'OBP', 'SLG', 'OPS', 'OPS+', 'HOF'] , encoding='latin-1', header = 0)


# classifications
y = data["HOF"]
print(y)

# features
x = data.drop('HOF', axis = 1)
x = x.drop('Player', axis = 1)

print(x)

knn_acc = 0
svmc_acc = 0
per_acc = 0

for i in range(100):
    x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size = 0.2)

    #x_train = x_train.reshape(-1, 1)
    #x_test = x_test.reshape(-1, 1)
    #y_train = y_train.reshape(1, -1)
    #y_test = y_test.reshape(-1, 1)

    knn = KNeighborsClassifier(n_neighbors=7)
    svmc = svm.SVC()
    per = Perceptron()
        
    svmc.fit(x_train, y_train)
    knn.fit(x_train, y_train)
    per.fit(x_train, y_train)
        
    # Predict on dataset which model has not seen before

    predicted = knn.predict(x_test)
    knn_acc += accuracy_score(y_test, predicted)

    predicted = svmc.predict(x_test)
    svmc_acc += accuracy_score(y_test, predicted)

    predicted = per.predict(x_test)
    per_acc += accuracy_score(y_test, predicted)

knn_acc = knn_acc / 100
svmc_acc = svmc_acc / 100
per_acc = per_acc / 100

print("knn_acc: " + str(knn_acc))
print("svmc_acc: " + str(svmc_acc))
print("per_acc: " + str(per_acc))


#print(features)
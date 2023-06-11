# -*- coding: utf-8 -*-
"""MultiClassClassification_MNIST

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fbbiT_VWFu1bmoSlwr8X-Yh9ErBXEm0r
"""

import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#load the dataset
# x train is training images, y train is train labels
(x_train,y_train), (x_test,y_test) = mnist.load_data()

#converting all images from 2d array to 1d array by using -1
x_train = x_train.reshape((x_train.shape[0],-1))
x_test = x_test.reshape((x_test.shape[0],-1))

#normalize
x_train=x_train/255.0
x_test = x_test/255.0

pca = PCA(n_components=2)
extracted_features = pca.fit_transform(x_train)

print(extracted_features)

model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

extracted_features_test = pca.transform(x_test)
y_pred = model.predict(x_test)

performance = classification_report(y_test,y_pred)
print(performance)

matrix = confusion_matrix(y_test,y_pred)
print(matrix)

#visualise the principal components

plt.figure(figsize=(10,8))
for digit in range(10):# here we have 10 classes, which are 0 to 9
  mask = (y_train == digit)
  plt.scatter(extracted_features[mask,0],extracted_features[mask,1],label=str(digit),alpha=0.5)

plt.xlabel("Principal component 1")
plt.ylabel("principla component 2")
plt.legend() # gives each class a different color in graph
plt.title("mnist-Principal Components")
plt.show()
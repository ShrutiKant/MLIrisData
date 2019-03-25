# -*- coding: utf-8 -*-
"""
Iris dataset for classification.
Problem: Predict the class of the flower based on available attributes.
"""

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns                  #it uses matplotlib to draw plots.

#Importing dataset
dataset = pd.read_csv('iris_data.csv')

# Data visualization
# We would like to see how the three species are different in terms of distribution of these four numerical attributes.For that, we analyse their histograms.

sns.set(style = "darkgrid")

#Visualize data for distribution of numerical values by species
sns.pairplot(hue = 'Class', data = dataset, size = 2.5)

#Create matrix of feature(independent var.) and dependent variable vector
X = dataset.iloc[:, :-1].values
y= dataset.iloc[:, 4].values

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Fitting the classification model on training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

#Predicting the Test set results
y_pred = classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("Accuracy: ", accuracy_score(y_test, y_pred))










#Encoding categorical variable
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)

#onehotencoder = OneHotEncoder()
#y = onehotencoder.fit_transform(y[:, np.newaxis]).toarray()
#Avoiding dummy variable trap
#y =y[:, 1:]
    
#A = y.flatten()
#y = np.reshape(y, -1)
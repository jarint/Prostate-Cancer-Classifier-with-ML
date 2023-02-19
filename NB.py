# -*- coding: utf-8 -*-
"""
@author: Jarin
"""

import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:\\Users\\Jarin\\Desktop\\Final Project - MDSC\\Prostate_Cancer.csv")

data.info()
data.drop(['id'], axis = 1)

def diagnosis_value(diagnosis_result):
    if diagnosis_result == 'M':
        return 1
    else:
        return 0
data['diagnosis_result'] = data['diagnosis_result'].apply(diagnosis_value)
x = np.array(data.iloc[:, 1:])
y = np.array(data['diagnosis_result'])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.33, random_state = 42)


# importing the module of the machine learning model
from sklearn.naive_bayes import GaussianNB

# initializing the classifier
gnb = GaussianNB()

# training the classifier
model = gnb.fit(x_train, y_train)

# making the predictions
predictions = gnb.predict(x_test)

# printing the predictions
print(predictions)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

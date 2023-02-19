# -*- coding: utf-8 -*-
"""
@author: Jarin
"""

import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:\\Users\\Jarin\\Desktop\\Final Project - MDSC\\Prostate_Cancer.csv")

print(data.shape)

df = pd.DataFrame(data=data)
print(df.head(5))
data.drop(['id'], axis = 1)
def diagnosis_value(diagnosis_result):
    if diagnosis_result == 'M':
        return 1
    else:
        return 0

data['diagnosis_result'] = data['diagnosis_result'].apply(diagnosis_value)

x = np.array(data.iloc[:, 2:])
y = np.array(data['diagnosis_result'])

from sklearn.model_selection import train_test_split
#training_set, test_set = train_test_split(df, test_size = 0.3, random_state = 1)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.33, random_state = 42)


from sklearn.svm import SVC
classifier = SVC(C=1,kernel='poly', random_state = 1, degree = 4)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
accuracy = float(cm.diagonal().sum()) / len(y_test)
print("\nAccuracy of SVM for the given dataset: ", accuracy*100)
#
#
#
#
#
#

#from sklearn.model_selection import StratifiedShuffleSplit
#from sklearn.model_selection import GridSearchCV
#C_range = np.logspace(-2, 10, 13)
#gamma_range = np.logspace(-9, 3, 13)
#param_grid = dict(gamma=gamma_range, C=C_range)
#cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
#grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
#grid.fit(x, y)
#
#print("The best parameters are %s with a score of %0.2f"
#      % (grid.best_params_, grid.best_score_))
#
#C_2d_range = [1e-2, 1, 1e2]
#gamma_2d_range = [1e-1, 1, 1e1]
#classifiers = []
#for C in C_2d_range:
#    for gamma in gamma_2d_range:
#        clf = SVC(C=C, gamma=gamma)
#        clf.fit(x_test, y_test)
#        classifiers.append((C, gamma, clf))
#
#

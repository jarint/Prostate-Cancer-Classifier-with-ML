# -*- coding: utf-8 -*-
"""
@author: Jarin
"""

import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
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


#sns.lmplot(x = 'radius', y = 'texture', hue = 'diagnosis_result', data = data)
#sns.lmplot(x = 'radius', y = 'perimeter', hue = 'diagnosis_result', data = data)
#sns.lmplot(x = 'radius', y = 'symmetry', hue = 'diagnosis_result', data = data)
#sns.lmplot(x = 'radius', y = 'area', hue = 'diagnosis_result', data = data)
#sns.lmplot(x = 'radius', y = 'compactness', hue = 'diagnosis_result', data = data)
#sns.lmplot(x = 'radius', y = 'smoothness', hue = 'diagnosis_result', data = data)
#sns.lmplot(x = 'radius', y = 'fractal_dimension', hue = 'diagnosis_result', data = data)
#sns.lmplot(x ='smoothness', y = 'compactness', hue = 'diagnosis_result', data = data)


x = np.array(data.iloc[:, 1:])
y = np.array(data['diagnosis_result'])


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.33, random_state = 42)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 23)
print(knn.fit(x_train, y_train))

print(knn.score(x_test, y_test))


neighbors = []
cv_scores = []

from sklearn.model_selection import cross_val_score
# perform 10 fold cross validation
for k in range(1, 51, 2):
    neighbors.append(k)
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(
        knn, x_train, y_train, cv = 10, scoring = 'accuracy')
    cv_scores.append(scores.mean())


MSE = [1-x for x in cv_scores]

# determining the best k
optimal_k = neighbors[MSE.index(min(MSE))]
print('The optimal number of neighbors is % d ' % optimal_k)

# plot misclassification error versus k
plt.figure(figsize = (10, 6))
plt.plot(neighbors, MSE)
plt.xlabel('Number of neighbors')
plt.ylabel('Misclassification Error')
plt.show()

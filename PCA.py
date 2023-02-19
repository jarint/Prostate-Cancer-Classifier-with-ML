# -*- coding: utf-8 -*-
"""
@author: Jarin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix


df = pd.read_csv("C:\\Users\\Jarin\\Desktop\\Final Project - MDSC\\Prostate_Cancer.csv")
display(df.head())

df = df.drop('id', axis = 1)

for i in range(len(df)):
    if df.iloc[i]['diagnosis_result'] == 'M':
        df.at[i, 'diagnosis_result'] = 1
    else:
        df.at[i, 'diagnosis_result'] = 0

df = df.fillna(df.mean())
display(df.head())

X = df.drop('diagnosis_result', axis = 1)
y = df['diagnosis_result']

y.value_counts().plot(kind = 'bar')
plt.xticks([0,1], ['Malignant', 'Benign'])
plt.ylabel('Count')

scaler = StandardScaler()
X = scaler.fit_transform(X)

pca = PCA(n_components=8)
pca.fit(X)
X_pca = pca.transform(X)

cum_exp_var = []
var_exp = 0
for i in pca.explained_variance_ratio_:
    var_exp += i
    cum_exp_var.append(var_exp)

fig, ax = plt.subplots(figsize=(8,6))
ax.bar(range(1,9), cum_exp_var)
ax.set_xlabel('# Principal Components')
ax.set_ylabel('% Cumulative Variance Explained')

train_f1 = []
test_f1 = []

for i in range(8):

    X = X_pca[:,0:i+1]

    # Train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=42)

    # Perform feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Fit k-NN classifier and make predictions
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)
    train_f1.append(f1_score(y_train, y_pred_train))
    test_f1.append(f1_score(y_test, y_pred_test))

# Plot accuracy by n_neighbors
plt.figure(figsize=(8, 6))
plt.plot(range(1,9), train_f1, label='Train f1 Score')
plt.plot(range(1,9), test_f1, label='Test f1 Score')
plt.ylabel('f1 Score')
plt.xlabel('# of Principal Components')
plt.legend()
plt.show()

X = X_pca[:,0:4]

# Train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=42)

# Perform feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit k-NN classifier and make predictions
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)

print(f'Train f1 Score: {f1_score(y_train, y_pred_train)}')
print(f'Test f1 Score: {f1_score(y_test, y_pred_test)}')
print(classification_report(y_test, y_pred_test))

#random
# Define X and y

X = df.drop('diagnosis_result', axis=1)
y = df['diagnosis_result']

# Loop to select random columns to be used in classifier
random_cols = []

for i in range(3):
    rand_col = 'X' + str(random.randint(1,9))
    random_cols.append(rand_col)

print(random_cols)

X = X[random_cols]

# Train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=42)

# Perform feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit k-NN classifier and make predictions
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)

print(f'Train f1 Score: {f1_score(y_train, y_pred_train)}')
print(f'Test f1 Score: {f1_score(y_test, y_pred_test)}')
print(classification_report(y_test, y_pred_test))

# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 21:21:24 2018

@author: cijo
"""

import pandas as pd
from sklearn import svm
import timeit
from sklearn.cross_validation import train_test_split
import numpy as np

import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

data = pd.read_csv('hepatitis_csv.csv')
data.head()

# preparing data
data = data.dropna()
data_d = pd.get_dummies(data)

x = data_d.loc[:, 'age':'varices_True'].values # INPUT DATA READY

## PREPARING OUTPUT ARRAY
output_data = data['class']
output_data = output_data.values
# if the input is Y then change to 1 else change to -1
y = [1 if i == 'live' else -1 for i in output_data] 


# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

seed = 7
results = {
        'accuracy': [],
        'average_precision': [],
        'precision': [],
        'recall': []
}
names = []
scorings = ['accuracy', 'average_precision', 'precision', 'recall']

for scoring in scorings: 
    print(scoring)
    print("*" * 10)
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=scoring)
        results[scoring].append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        

# accuracy chart
fig = plt.figure()
fig.suptitle('Accuracy Comparison')
ax = fig.add_subplot(111)      
plt.boxplot(results['accuracy'])
ax.set_xticklabels(names)

# precision chart
fig = plt.figure()
fig.suptitle('Precision Comparison')
ax = fig.add_subplot(111)      
plt.boxplot(results['precision'])
ax.set_xticklabels(names)
plt.show()

#average precision
fig = plt.figure()
fig.suptitle('Average Precision Comparison')
ax = fig.add_subplot(111)      
plt.boxplot(results['average_precision'])
ax.set_xticklabels(names)

#recall
fig = plt.figure()
fig.suptitle('Recall Comparison')
ax = fig.add_subplot(111)      
plt.boxplot(results['recall'])
ax.set_xticklabels(names)
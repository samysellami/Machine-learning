#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Khan
"""

"""
In this problem, you will implement LOGISTIC REGRESSION,
and SUPPORT VECTOR MACHINES (default values for all attributes) using sklearn 
to identify three different types of irises (Setosa, Versicolour, virginica)
using sepal length and sepal width as features. 

For evaluation, you will use K-fold Corss Validation -- this part you will implement yourself without any libraries

"""
#TO DO --------------------- Import all libraries here --------------------------

from sklearn import datasets


""" Loading Iris dataset  
Reading the sepal width and petal length
information for all rows 
X contains the features, y contains the targets"""
iris = datasets.load_iris()
X, y = iris.data[:, 0:2], iris.target



#TO DO ---- 2 POINTS --------------------- Create the two classifier objects (Default values for all attributes) ---------------------



print('5-fold cross validation:\n')

#TO DO ---- 8 POINTS --------------------- Mode Fitting via Cross Validation ---------------

""" Fit both models using k-fold cross validation as follows 

--- 1 Point --- Get the value of k from user




--- 3 Points --- Create splits for cross-validation as per the value of k -- DO NOT USE ANY LIBRARIES




---- 4 Points --- Then use those splits to fit the models, calculate and print the mean classification 
accuracy and its standard deviation for each classifier as follows

Accuracy: VAL (+/- VAL) [Logistic Regression]
Accuracy: VAL (+/- VAL) [SVM]

"""



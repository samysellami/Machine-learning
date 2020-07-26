"""
@author: Khan
"""

##The Problem is to compare the performance of Decision Tree classifier 
# implementation for Entropy and Gini.
# Your modeled Decision Tree will compare the new records metrics (provided) with 
# the prior records (training data) that correctly classified the balance 
# scale’s tip direction.


## Import Important Packages
import numpy as np
import pandas as pd




# Import Data
# For importing the data and manipulating it, we are going to use pandas dataframes.
# After downloading the data file, you'r required to use Pandas read_csv() method to import data into 
# pandas dataframe. Since our data is separated by commas “,” and there is no header in our 
# data, so we will put header parameter’s value “None” and sep parameter’s value as  “,”.
Data = pd.read_csv(
'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',
                           sep= ',', header= None)
print ('Dataset Lenght:: ', len(Data))
print ('Dataset Shape:: ', Data.shape)
print ('Dataset:: ')
Data.head()


## Now divides data into feature set & target set (Slicing), where
## X is your Data matrix and Y is your Class Representation

# ******** 3 Poitns *********** Your Code Goes Here ***********

X = 
Y = 
# Now Split data into train and test set. Test dataset should not be mixed up while building model. 
# Even during standardization, you should not standardize the test set.
X_train, X_test, y_train, y_test = 



# Now you'll fit Decision tree algorithm on training data, predicting labels for
# validation dataset and printing the accuracy of the model using various parameters
# such as "class_weight = None", "criterion = gini and entropy", "max_depth = 3",
# "max_features = None", "max_leaf_nodes = None', "min_samples_leaf = 5", 
# "min_samples_split = 2", "min_weight_fraction_leaf = 0.0", "presort = False", 
# "random_state = 100", and "splitter = 'best'"

# ******** 4 Poitns *********** Your Code Goes Here ***********


# Gini
clf_gini = 
clf_gini.fit(X_train, y_train)


## Entropy
clf_entropy = 
clf_entropy.fit(X_train, y_train)


## Now, you've modeled 2 classifiers. One classifier with gini index & 
# one with information gain as the criterion. You are ready to predict 
# classes for our test set. You can use predict() method. Try to predict 
# target variable for test set’s records.

# ******** 3 Poitns *********** Your Code Goes Here ***********


## Predict with Gini Index
y_pred = 

# Prediction with information gain
y_pred_en = 


# Evaluate the Classifier for Gini and Entorpy
print ('Accuracy for Gini ', accuracy_score(y_test,y_pred)*100)
print ('ccuracy for IG ', accuracy_score(y_test,y_pred_en)*100)


import pandas as pd
import numpy as np
from sklearn import datasets
import math
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
import keras
from tensorflow.python.framework import ops
import cv2
from numpy.random import seed

##########################  SVM   ##############################################################
from sklearn.svm import SVC
SVC()
#################################################################################################




##############################  Decision trees ##################################################
from sklearn.tree import DecisionTreeClassifier
DecisionTreeClassifier()

from sklearn.tree import DecisionTreeRegressor
DecisionTreeRegressor()
#################################################################################################




############################## AdaboostClassifier and Randam forest  ############################
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(base_estimator=algo, n_estimators=10)
model.estimator_weights_
model.estimator_

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state=42)
##################################################################################################



###############################  Cross validation ############################################
from sklearn.cross_validation import cross_val_score
score= score.mean(cross_val_score(regression_tree, X, y , scoring='mean_squared_error', cv=crossvalidation, n_jobs=1))
###############################################################################################"





###############################  Scaling ######################################################
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
###############################################################################################




########################   Reshaping  ###########################################################
from numpy import reshape
y=y.reshape((y.shape[0], 1))
#################################################################################################




################################ Splitting data ################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2)
################################################################################################




###################    Logistic Regression  ####################################################
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train, y_train)
y_predict=model.predict(X_test)
#################################################################################################




###################  Reading an argument  ######################################################
person = input('Enter your name: ')
################################################################################################



####################    Accuracy  ################################################################
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

accuracy_score()
print("Precision = ",precision_score(y_test.values, y_predict, average = None))
print("Recall = ", recall_score(y_test.values, y_predict, average=None))

from sklearn.metrics import mean_squared_error
y_predict = regression_model.predict(X_test)
regression_model_mse = mean_squared_error(y_predict, y_test)
####################################################################################################





###########################   Ploting ############################################################
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error rate k value')
plt.xlabel('Kvalue')
plt.ylabel('Mean error')

_, axes = plt.subplots(1, 2, figsize=(11,5), sharey=True)

axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.cm.Spectral)
axes[0].set_xlim((-1.5, 2.5))
axes[0].set_ylim((-1, 1.5))

test_x1 = np.linspace(-1.5, 2.5, 20)
test_x2 = np.linspace(-1,   1.5, 20)
for x1 in test_x1:
    for x2 in test_x2:
        y = predict([[x1, x2]])
        color = 'blue' if y > 0.5 else 'red'
        axes[1].scatter(x1, x2, c=color)
plt.show()
#######################################################################################################

################################   Bootstrap ############################################################
from sklearn.utils import resample
aa, bb= resample(a, b)
########################################################################################################


############################################# CNN  ######################################################
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10,10)
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
######################################################################################################




################################## Miscellaneous #######################################################

np.where(alphas > thresh)[0]

print('prediction for {} is {}'.format(X_test, y_pred))
y_test.reset_index(drop=True, inplace=True)

df = pd.read_csv('ks-projects-201801.csv')
df=df[df[df.columns[9]]!='canceled']

df=pd.DataFrame(df)

y= df[df.columns[9]]

X = df.loc[:,['backers']]

val, counts = np.unique(x, return_counts=True)

for p, v in zip(freqs, val):
    res -= p * entropy(y[x == v])

isinstance(v, dict):
{c: (a == c).nonzero()[0] for c in np.unique(a)}

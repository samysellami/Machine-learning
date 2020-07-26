# This question is about implementing a simple ANN for a Regression Task 
# Use Keras
# Don't do any additional imports, everything is already there


import numpy as np
import keras
from keras import models
from keras import layers
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Setting random seed
np.random.seed(0)

# Generating features matrix and target vector
# Number of samples = 1000, Number of features = 3
features, target = make_regression(n_samples = 10000,
                                   n_features = 3,
                                   n_informative = 3,
                                   n_targets = 1,
                                   noise = 0.0,
                                   random_state = 0)

#Divding data into training and test sets 
features_train, features_test, target_train, target_test = train_test_split(
features, target, test_size=0.33, random_state=0)


#TO DO ---- 10 POINTS ---------- Create your ANN, Compile it, and Train it ---------------------
""" First Layer (Dense, neurons = 32, Activation = relu)
    Second Layer (Dense, neurons = 32, Activation = relu)
    Third Layer (Dense, neurons = DECIDE YOURSELF, Activation = None)
    Loss (MSE), Optimizer (Adam), Metrics (MSE)
    Epochs (30), Batch Size (100)
"""


# TO DO ---- 1POINT ---- Start neural network
model = models.Sequential()


# TO DO ---- 2 POINTS ---- Add First Layer
model.add(layers.Dense(32, activation='relu'))


# TO DO ---- 2 POINTS ---- Add Second Layer
model.add(layers.Dense(32, activation='relu'))


# TO DO ---- 2 POINTS ---- Add Third Layer, You must decide the number of neurons yourself for this layer
model.add(layers.Dense(32, activation='relu'))
#we conserve the same number to reduce overfitting

# TO DO ---- 2 POINTS ---- Compile neural network
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['MSE'])

# TO DO ---- 2 POINTS ---- Train neural network
batch_size=100
epochs=30
model.fit(features_train, target_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(features_test, target_test))


#TO DO ---- 5 POINTS ---------- Plot Training and Test Loss ---------------------
score = model.evaluate(feature_test, target_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

predicted_classes = model.predict_classes(features_test)



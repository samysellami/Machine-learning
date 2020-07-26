


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier



#function to calculate alphas 
def get_alphas(weights, n_estimators, X, y, y_predict):
    alphas=[1/len(X) for i in range(len(X))]
    for i in range(n_estimators):
        for j in range(len(alphas)):
            if y_predict[j]==y[j]:
                alphas[j]=alphas[j]*np.exp(-weights[i])
            else:
                alphas[j]=alphas[j]*np.exp(weights[i])
    return alphas


#funcion to predict if a country is in europe
def predict(Country, dataset, y_predict):
    for i in range(len(dataset)):
        if dataset['Country'].iloc[i]==Country:
            return y_predict[i]
    if i==(len(dataset)-1):
        print('no such country in dataset(it could be an outlier!!)')    
    #y_predict= model.predict(dataset[dataset['Country']==Country].iloc[0,1:-1])


    #read dataset
dataset = pd.read_csv('countries.csv')
print(dataset)


print(dataset.columns)

#get dummy variables
dataset = pd.get_dummies(dataset, columns=['Region'])
print(dataset)

#drop the columns that we do not want 
dataset=dataset.drop(columns=['Region_ASIA', 'Region_AMERICA', 'Region_OCEANIA', 'Region_AFRICA'])
print(dataset)


dataset=dataset.dropna()
dataset.reset_index(drop=True, inplace=True)

#selecting features and label
X=dataset.drop(columns=['Country','Region_EUROPE'])
y=dataset['Region_EUROPE']
print(X,y)
print(X.shape)
print(y.shape)


#performing the classification
n_estimators=10
algo= LogisticRegression()
model= AdaBoostClassifier(base_estimator= algo, n_estimators=n_estimators, algorithm='SAMME')

model.fit(X,y)
weights= model.estimator_weights_
print (weights)
model.estimators_[1]


y_predict=model.predict(X)
print(y_predict)


#accuracy
from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y, y_predict)
print(accuracy)

from sklearn.model_selection import cross_val_score
print(cross_val_score(model, X, y, cv=3).mean())  

#obtaining alphas
alphas= get_alphas(weights, n_estimators, X, y, y_predict)
print(alphas)

index=[ i for i in range(len(alphas))  if alphas[i]<0.001]
index

#shrinking X and y (without outliers)
y=y[index]
X=X.iloc[index,:]
dataset=dataset.iloc[index]


y.reset_index(drop=True, inplace=True)
X.reset_index(drop=True, inplace=True)
dataset.reset_index(drop=True, inplace=True)

#fitting the dataset again
model.fit(X,y)
weights= model.estimator_weights_
print(weights)

y_predict=model.predict(X)
print(y_predict)

#accuracy
accuracy= accuracy_score(y, y_predict)
print(accuracy)


from sklearn.model_selection import cross_val_score
print(cross_val_score(model, X, y, cv=3).mean())  

#predicting 
Country='Spain '
predict(Country, dataset, y_predict)



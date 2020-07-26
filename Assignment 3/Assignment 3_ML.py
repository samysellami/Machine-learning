
# coding: utf-8

# Exercise:

# In[386]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

### 1 GENERATE DATA
iris = datasets.load_iris()


# In[387]:


iris


# In[388]:


X = iris.data
y = iris.target


# In[389]:


X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2)


# In[243]:


X_train.shape


# In[242]:


y_train


# In[11]:


model=LogisticRegression()


# In[14]:


model.fit(X_train, y_train)


# In[18]:


y_pred= model.predict(X_test)
print(X_test, y_pred)


# In[57]:


accuracy=accuracy_score(y_pred, y_test)
print(accuracy)


# HomeTask1: Finalizing Gradient Descent for Logistic Regression

# In[390]:


def sigmoid(z):
    return np.exp(z)/(1+ np.exp(z)) 


# In[391]:


def gradient_LR(x, y, alpha):

    w=np.array([-1.0,1.0])
    N=len(x)
    iteration=1
    
    while iteration<1000:    
        z= w[0] + w[1]*x
        delta= sigmoid(z) - y
        
        db0 = np.sum(delta)/N
        db1 = np.dot(x.T,delta)/N
        
        w[1] = w[1] - alpha*db1
        w[0] = w[0] - alpha*db0

        iteration+=1
        
    return w


# In[393]:


X= np.array([iris.data[i,:1] for i in range(len(iris.data)) if iris.target[i]!=2])
y= np.array([iris.target[i] for i in range(len(iris.data)) if iris.target[i]!=2])


# In[394]:


from numpy import reshape
y=y.reshape((y.shape[0], 1))


# In[401]:


X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=1)


# In[396]:


y.shape


# In[402]:


b0, b1= gradient_LR(X_train, y_train, 0.01)
print(b0, b1)


# In[398]:


y_pred=[int(sigmoid(b0+ b1*x)>0.5) for x in X_test]
y_pred


# In[399]:


plt.scatter(X_test,y_test)
plt.scatter(X_test, y_pred)
plt.show


# In[403]:


accuracy=accuracy_score(y_pred, y_test)
print(accuracy)


# HOMEWORK N°3

# In[405]:


import pandas as pd
df = pd.read_csv('ks-projects-201801.csv')


# In[406]:


df.columns
df.dtypes


# In[407]:


df


# In[408]:


df=df[df[df.columns[9]]!='canceled']


# In[409]:


df=pd.DataFrame(df)
print(df)


# In[410]:


#y=df.iloc[:,9:10]
y= df[df.columns[9]]
#y = df.loc[:,['state']]
#print(y)


# In[411]:


#X = df.drop(df.columns[9], axis=1)
#X= df[df.columns[10]]
#X=df['pledged']
X = df.loc[:,['backers']]
#print(X)


# In[412]:


X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2)


# In[413]:


y_test.reset_index(drop=True, inplace=True)


# In[414]:


model=LogisticRegression()


# In[415]:


model.fit(X_train, y_train)


# In[416]:


y_predict=model.predict(X_test)


# In[417]:


from sklearn.metrics import precision_score

print("Precision = ",precision_score(y_test.values, y_predict, average = None))
print("Recall = ", recall_score(y_test.values, y_predict, average=None))


# In[418]:


TP=0
FP=0
FN=0
for i in range(len(y_test)):
    if y_predict[i]=='successful':
        if y_predict[i]==y_test[i]:
            TP+=1
        else:
            FP+=1
    else:
        if y_predict[i]!=y_test[i]:
            FN+=1

precision = TP/(TP + FP)
recall = TP/(TP + FN)


# In[419]:


print(recall)
print(precision)


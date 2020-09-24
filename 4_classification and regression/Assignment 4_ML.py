
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import random 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=12, 10


# In[14]:


x=np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)
y=np.sin(x) + np.random.normal(0, 0.15, len(x))
data= pd.DataFrame(np.column_stack([x,y]), columns=['x', 'y'])
plt.plot(data['x'], data['y'], '.')


# In[15]:


for i in range (2,16):
    colname='x_%d'%i
    data[colname]=data['x']**i


# In[7]:


data


# In[20]:


from sklearn.linear_model import LinearRegression
def linear_regression(data, power, models_to_plot):
    predictors=['x']
    if power >=2:
        predictors.extend(['x_%d'%i for i in range(2, power+1)])
        
    linreg= LinearRegression(normalize=True)
    linreg.fit(data[predictors], data['y'])
    y_pred= linreg.predict(data[predictors])
    
    if power in models_to_plot:
        plt.subplot(models_to_plot[power])
        plt.tight_layout()
        plt.plot(data['x'], y_pred)
        plt.plot(data['x'], data['y'], '.')
        plt.title('plot for power: %d'%power)


# In[21]:


models_to_plot={1:231, 3:232, 6:233, 9:234, 12:235, 15:236}
for i in range(1,16):
    linear_regression(data, power=i, models_to_plot=models_to_plot)


# In[22]:


from sklearn.linear_model import Ridge
def ridge_regression(data, predictors, alpha, models_to_plot={}):

    ridgereg= Ridge(alpha=alpha, normalize=True)
    ridgereg.fit(data[predictors], data['y'])
    y_pred= ridgereg.predict(data[predictors])
    
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'], y_pred)
        plt.plot(data['x'], data['y'], '.')
        plt.title('plot for alpha: %.3g'%alpha)


# In[23]:


predictors=['x']
predictors.extend(['x_%d'%i for i in range(2, 16)])
alpha_ridge=[1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]
models_to_plot={1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
for i in range(10):
    ridge_regression(data, predictors, alpha_ridge[i], models_to_plot)


# In[24]:


from sklearn.linear_model import Lasso
def lasso_regression(data, predictors, alpha, models_to_plot={}):

    lassoreg= Lasso(alpha=alpha, normalize=True, max_iter= 1e5)
    lassoreg.fit(data[predictors], data['y'])
    y_pred= lassoreg.predict(data[predictors])
    
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'], y_pred)
        plt.plot(data['x'], data['y'], '.')
        plt.title('plot for alpha: %.3g'%alpha)


# In[25]:


predictors=['x']
predictors.extend(['x_%d'%i for i in range(2, 16)])
alpha_lasso=[1e-15, 1e-10, 1e-8,1e-5, 1e-4, 1e-3, 1e-2, 1, 5, 10]
models_to_plot={1e-10:231, 1e-5:232, 1e-4:233, 1e-3:234, 1e-2:235, 1:236}
for i in range(10):
    lasso_regression(data, predictors, alpha_lasso[i], models_to_plot)


# In[78]:


from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, Lasso, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error


# In[48]:


df=pd.read_csv('Hitters.csv').dropna().drop('Player', axis=1)
dummies=pd.get_dummies(df[['League', 'Division', 'NewLeague']])


# In[67]:


y=df.Salary
X_=df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
X=pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)


# In[81]:


alphas= 10**np.linspace(10, -2, 100)*0.5
ridge=Ridge(normalize=True)
coefs=[]
for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
np.shape(coefs)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3)


# KNN CLASSIFICATION

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[90]:


url="http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"


# In[91]:


names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']


# In[92]:


dataset = pd.read_csv(url, names=names)


# In[93]:


dataset


# In[124]:


X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 4].values


# In[135]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3)


# In[137]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[138]:


X_train.astype('float64')


# In[139]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(algorithm= 'auto', leaf_size=30, metric='minkowski', n_jobs=1, n_neighbors=5, p=2, weights='uniform')
classifier.fit(X_train, y_train)


# In[140]:


y_pred=classifier.predict(X_test)


# In[145]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[154]:


error=[]
for i in range(1,40):
    knn=KNeighborsClassifier(algorithm= 'auto', leaf_size=30, metric='minkowski', n_jobs=1, n_neighbors=i, p=2, weights='uniform')
    knn.fit(X_train, y_train)
    pred_i=knn.predict(X_test)
    error.append(np.mean(pred_i!=y_test))


# In[155]:


plt.figure(figsize=(12, 6))
plt.plot(range(1,40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error rate k value')
plt.xlabel('Kvalue')
plt.ylabel('Mean error')


# HOMEWORK 4: 

# In[8]:


df=pd.read_csv('winequality-red.csv', sep=';')


# In[9]:


df


# In[16]:


X=df.iloc[:, :-1].values
y=df.iloc[:, 11].values


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3)


# In[19]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[21]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(algorithm= 'auto', leaf_size=30, metric='minkowski', n_jobs=1, n_neighbors=5, p=2, weights='uniform')
classifier.fit(X_train, y_train)


# In[22]:


y_pred=classifier.predict(X_test)


# In[24]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[25]:


error=[]
for i in range(1,40):
    knn=KNeighborsClassifier(algorithm= 'auto', leaf_size=30, metric='minkowski', n_jobs=1, n_neighbors=i, p=2, weights='uniform')
    knn.fit(X_train, y_train)
    pred_i=knn.predict(X_test)
    error.append(np.mean(pred_i!=y_test))


# In[26]:


plt.figure(figsize=(12, 6))
plt.plot(range(1,40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error rate k value')
plt.xlabel('Kvalue')
plt.ylabel('Mean error')


# In[28]:


error=[]
for i in range(1,40):
    knn=KNeighborsClassifier(algorithm= 'auto', leaf_size=30, metric='manhattan', n_jobs=1, n_neighbors=i, weights='uniform')
    knn.fit(X_train, y_train)
    pred_i=knn.predict(X_test)
    error.append(np.mean(pred_i!=y_test))
    
plt.figure(figsize=(12, 6))
plt.plot(range(1,40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error rate k value')
plt.xlabel('Kvalue')
plt.ylabel('Mean error')


# In[30]:


error=[]
for i in range(1,40):
    knn=KNeighborsClassifier(algorithm= 'auto', leaf_size=30, metric='chebyshev', n_jobs=1, n_neighbors=i, weights='uniform')
    knn.fit(X_train, y_train)
    pred_i=knn.predict(X_test)
    error.append(np.mean(pred_i!=y_test))
    
plt.figure(figsize=(12, 6))
plt.plot(range(1,40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error rate k value')
plt.xlabel('Kvalue')
plt.ylabel('Mean error')


# In[40]:


from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, Lasso, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error


# In[35]:


alphas= 10**np.linspace(10, -2, 100)*0.5
ridgeCV=RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', normalize=True)
ridgeCV.fit(X_train, y_train)
ridgeCV.alpha_


# In[49]:


alphas= 10**np.linspace(10, -2, 100)*0.5
ridge=Ridge(alpha=ridgeCV.alpha_, normalize=True)
ridge.fit(X_train, y_train)
mean_squared_error(y_test, ridge.predict(X_test))


# In[50]:


#ridge.fit(X, y)
pd.Series(ridge.coef_, index=df.columns[:-1])


# In[51]:


alphas= 10**np.linspace(10, -2, 100)*0.5
lassoCV=LassoCV(alphas=alphas, cv=10,  max_iter=10000, normalize=True)
lassoCV.fit(X_train, y_train)

lasso=Lasso( max_iter=10000, normalize=True)
lasso.set_params(alpha=lassoCV.alpha_)
lasso.fit(X_train, y_train)
mean_squared_error(y_test, lasso.predict(X_test))


# In[52]:


pd.Series(lasso.coef_, index=df.columns[:-1])


# In[53]:


lasso.coef_


# In[54]:


lasso.predict(X_test)


# In[ ]:


LogisticRegression()


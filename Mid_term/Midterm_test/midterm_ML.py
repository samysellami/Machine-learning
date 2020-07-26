import pandas as pd
from sklearn.metrics import precision_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from numpy import reshape
y=y.reshape((y.shape[0], 1))


df = pd.read_csv('ks-projects-201801.csv')
df=df[df[df.columns[9]]!='canceled']

df=pd.DataFrame(df)

y= df[df.columns[9]]

X = df.loc[:,['backers']]

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2)
y_test.reset_index(drop=True, inplace=True)

model=LogisticRegression()

model.fit(X_train, y_train)

y_predict=model.predict(X_test)


print("Precision = ",precision_score(y_test.values, y_predict, average = None))
print("Recall = ", recall_score(y_test.values, y_predict, average=None))

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(algorithm= 'auto', leaf_size=30, metric='minkowski', n_jobs=1, n_neighbors=5, p=2, weights='uniform')
classifier.fit(X_train, y_train)

error = []
for i in range(1, 40):
    knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='chebyshev', n_jobs=1, n_neighbors=i,
                               weights='uniform')
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error rate k value')
plt.xlabel('Kvalue')
plt.ylabel('Mean error')
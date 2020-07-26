
# coding: utf-8

# Homework 2: Linear regression and gradient descent 

# In[36]:


import numpy as np

import pandas as pd
df = pd.read_csv('mpg.csv')


# In[37]:


df=df.drop('name', axis=1)


# In[38]:


print(df.loc[354])


# In[39]:


import numpy as np

df = df.replace('?', np.nan)
df = df.dropna()


# In[40]:


df['origin'].unique()


# In[41]:


df['origin'] = df['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})
df = pd.get_dummies(df, columns=['origin'])


# In[42]:


df


# In[43]:


X = df.drop('mpg', axis=1)
y = df[['mpg']]


# In[44]:


from sklearn.model_selection import train_test_split

# Split X and y on training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[46]:


from sklearn.linear_model import LinearRegression

regression_model = LinearRegression()
# feed the linear regression with the train data to obtain a model.
regression_model.fit(X_train, y_train)


# In[47]:



coefficients = []
# add beta0
coefficients.append(regression_model.intercept_[0]) 
equation = 'Y = ' + str(round(coefficients[0],2))
for i in range(X_train.shape[1]):
    coefficients.append(regression_model.coef_[0][i])
    equation += ' + ' + str(round(regression_model.coef_[0][i],2))+'X'+str(i+1)
    
# print result equation
print(equation)


# In[48]:


regression_model.score(X_test, y_test)


# In[49]:


from sklearn.metrics import mean_squared_error
import math

y_predict = regression_model.predict(X_test)
regression_model_mse = mean_squared_error(y_predict, y_test)

print(math.sqrt(regression_model_mse))


# In[50]:


regression_model.predict([[4, 121, 110, 2800, 15.4, 81, 0, 1, 0]])


# In[51]:


import matplotlib.pyplot as plt

projected_column = 'displacement'
max_x = np.int64(X_train[projected_column].max())
min_x = np.int64(X_train[projected_column].min())
x = list(X_train[projected_column])
y = list(y_train['mpg'])
m = coefficients[df.columns.get_loc(projected_column)+1]
b = coefficients[0]
# now we are going to plot the points and the model obtained
plt.scatter(x, y, color='blue')  # you can use test_data_X and test_data_Y instead.
plt.plot([min_x, max_x], [m*min_x + b, m*max_x + b], 'r')
plt.title('Fitted linear regression', fontsize=16)
plt.xlabel('x', fontsize=13)
plt.ylabel('y', fontsize=13)


# Gradient Descent:

# In[52]:


import numpy as np
import matplotlib.pyplot as plt


# In[63]:


def generate_data(b1, b0, size, x_range = (-10, 10), noise_mean = 0, noise_std = 1):
    """
	input:
	b1, b0 - true parameters of data
	size - size of data, numbers of samples
    x_range - tuple of (min, max) x-values
    noise_mean - noise mean value
    noise_std - noise standard deviation
	
	output:
	data_x, data_y - data features
	"""
    noise = np.random.normal(noise_mean, noise_std, size)
    # rnd_vals = np.random.rand(size)
    # data_x = np.random.choice(x_range[1]-x_range[0], size)
    data_x = np.linspace(start=x_range[0], stop=x_range[1], num=size)
    data_y = b1 * data_x + b0 + noise
	
    return data_x, data_y


# In[64]:


data_x, data_y = generate_data(2.5, -7, 100)


# In[55]:


data_x


# In[65]:


def predict(x, y):
	"""
	input:
	x, y - data features
	
	output:
	b1, b0 - predicted parameters of data
	"""
	mean_x = x.mean()
	mean_y = y.mean()

	b1 = np.dot(y - mean_y, x - mean_x) / np.dot(x - mean_x, x - mean_x)
	b0 = mean_y - b1*mean_x

	return b1, b0


# In[66]:


b1, b0=predict(data_x, data_y)
print(b0, b1)


# In[67]:


def animate(data_x, data_y, true_b1, true_b0, b1, b0, x_range = (-10,10), label="Least squares"):
	plt.scatter(data_x, data_y)
	plt.plot([x_range[0], x_range[1]], 
           [x_range[0]*true_b1 + true_b0, x_range[1]*true_b1 + true_b0], 
           c="r", linewidth=2, label="True")
	plt.plot([x_range[0], x_range[1]], 
           [x_range[0]*b1 + b0, x_range[1]*b1 + b0], 
           c="g", linewidth=2, label=label)
	plt.legend()
	plt.show()


# In[70]:


animate(data_x, data_y, 2.5, -7, b1, b0)


# In[71]:


def gradient(x, y, alpha):
    
    w=np.array([-1.0,1.0])
    N=len(x)
    iteration=1
    
    #new_w=np.array([1.0,1.0])
    #b1, b0 = 1, -1
    #error=1
    while iteration<1000:    
        #delta= -y + (b1*x + b0)
        delta= -y + (w[1]*x + w[0])

        db0 = 2*np.sum(delta)/N
        db1 = 2*np.dot(delta,x)/N
        
        #b1=b1 - alpha*db1
        #b0=b0 - alpha*db0
        w[1] = w[1] - alpha*db1
        w[0] = w[0] - alpha*db0
            
        #error=np.sum(abs(new_w - w))
        #w=new_w
        iteration+=1
        
    return w


# In[72]:


gradient(data_x, data_y, 0.01)


# In[178]:


def gradient_descent(x, y, lr=1e-2, N=1000):
	# b1, b0 = np.random.random(size=2)
	b1, b0 = 1, -1

	iteration = 0
	while iteration < N:
		delta = b1 * x + b0 - y
		db0 = 2 * np.sum(delta) / N
		db1 = 2 * np.dot(delta, x) / N

		b1 = b1 - lr * db1
		b0 = b0 - lr * db0
		iteration += 1

	return b1, b0


# In[22]:


gradient_descent(data_x, data_y)


import numpy as np
import pandas as pd
from statistics import mode

# code is based on this resource - http://gabrielelanaro.github.io/blog/2016/03/03/decision-trees.html

def partition(a):
    return {c: (a == c).nonzero()[0] for c in np.unique(a)}


def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float') / len(s)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res


def information_gain(y, x):

    res = entropy(y)

    # We partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float') / len(x)

    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])

    return res


def is_pure(s):
    return len(set(s)) == 1

def information_gain_ratio(y, x):
    return information_gain(y, x)/entropy(x)


def predict(tree, x , fields):
    for k,i in tree.items():
        #print(k,i)
        find=0
        for j in range(len(x)):
            if k == "{} = {}".format(fields[j], x[j]):
                find=1
                break
        
        if find==0:
            continue
        
        if isinstance(i, dict):
            return predict(i, x, fields)
            break
        else:
            return mode(i)

                
def preprocess_cont_data(data, columns):
    subset=dataset[columns]
    subset =subset.dropna()
    subset= subset.sort_values(by=[columns[0]])
    subset_x=subset.iloc[:, 0].values
    subset_y=subset.iloc[:, 1].values
    return subset_x, subset_y



def get_threshold(x, y):
    t=[]
    for i in range(len(x)-1):
        t.append((x[i]+x[i+1])/2)
        
    gain=0
    max_gain=0
    t_best=0
    for thresh in t:
        p1=len(x[x<thresh])/len(x)
        p2=len(x[x>thresh])/len(x)
        gain=entropy(y) - p1*entropy(y[x<thresh]) - p2*entropy(y[x>thresh])
        if gain> max_gain:
            max_gain=gain
            t_best=thresh
    return t_best



def transform_data(x, threshold):
    for i in range (len(x[:, 2])):
        if x[i, 2] !='nan':
            if x[i, 2]<threshold:
               x[i, 2]=0 
    for i in range (len(x[:, 2])):
        if x[i, 2] !='nan':
            if x[i, 2]>=threshold:
                x[i, 2]=1 
    return x


def recursive_split_45(x, y, fields):
    # If there could be no split, just return the original set
    if is_pure(y) or len(y) == 10: # we set the minimum length of the leaf to 10 (pre-prunning)
        return y

    # We get attribute that gives the highest information gain
    gain = np.array([information_gain_ratio(y, x_attr) for x_attr in x.T]) ## we use the information gain ratio
    selected_attr = np.argmax(gain)

    # If there's no gain at all, nothing has to be done, just return the original set
    if np.all(gain < 1e-6):
        return y

    # We split using the selected attribute
    sets = partition(x[:, selected_attr])

    res = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)

        res["{} = {}".format(fields[selected_attr], k)] = recursive_split(
            x_subset, y_subset, fields)

    return res





pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

## we upload titanic dataset
dataset = pd.read_csv('titanic_modified.csv')
dataset['Embarked']=dataset['Embarked'].astype(str)

## we preprocess the real values 'Age'
subset_x, subset_y= preprocess_cont_data(dataset, ['Age', 'Survived'])

# we calculate the threshold for the 'Age' datas
threshold=get_threshold(subset_x, subset_y)

dataset=dataset.dropna()
x = dataset.iloc[:, :6].values
y = dataset.iloc[:, 6].values
fields = list(dataset.columns.values[0:6])

## we transform the continuous features in classified ones
x=transform_data(x,threshold)

## we build the tree and print it 
tree=recursive_split_45(x, y, fields)
print(tree)

## We predict one data from our dataset
i=18
g=predict(tree, x[i,:], fields)
print('the prediction is:',g)





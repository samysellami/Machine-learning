
# coding: utf-8

# In[708]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from matplotlib import markers
import itertools


# In[709]:


dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
#X=np.unique(X, axis=0)
markers = markers.MarkerStyle.markers
colormap = plt.cm.Dark2.colors


# In[710]:


distances = squareform(pdist(X, metric='euclidean'))
np.fill_diagonal(distances, np.inf)


# In[711]:


num_points = X.shape[0]
X_data = np.c_[X, np.arange(0, num_points, 1)]


# In[712]:


X_data.shape


# In[713]:


clusters = np.c_[distances, np.zeros((num_points, 2))]


# In[714]:


dist_col_id = num_points
clust_col_id = num_points + 1


# In[715]:


clusters[:, clust_col_id] = np.argmin(clusters[:, :num_points], axis=1)
clusters[:, dist_col_id] = np.amin(clusters[:, :num_points], axis=1)


# In[716]:


def find_clusters_to_merge(clusters, dist_col_id, clust_col_id):
    c1_ind=np.argmin(clusters[:,dist_col_id])
    c2_ind=clusters[c1_ind, clust_col_id].astype(int)
    distance=clusters[c1_ind, dist_col_id]
    return c1_ind, c2_ind, distance


# In[717]:


def single_link_merge(c1_ind, c2_ind, X_data, clusters, dist_col_id, clust_col_id):
    num_points = clusters.shape[0]

    if c1_ind>c2_ind:
        temp=c1_ind
        c1_ind=c2_ind
        c2_ind=temp
    
    #print(c1_ind, c2_ind)
    for i in range(num_points):
        if clusters[i, c1_ind] > clusters[i, c2_ind]:
            clusters[i, c1_ind]=clusters[i, c2_ind]
            
    for j in range (num_points):
        if clusters[c1_ind, j] > clusters[c2_ind, j]:
            clusters[c1_ind, j]=clusters[c2_ind, j]
            
    clusters=np.delete(clusters, c2_ind, 1)
    clusters=np.delete(clusters, c2_ind, 0)
    np.fill_diagonal(clusters, np.inf)
    num_points = clusters.shape[0]

    for i in range(len(X_data[:,2])):
        if X_data[i, 2]==c2_ind:
            X_data[i, 2]=c1_ind
        if X_data[i,2]>c2_ind:
            X_data[i,2]=X_data[i,2]-1
        
    dist_col_id-=1
    clust_col_id-=1
    
    clusters[:, clust_col_id] = np.argmin(clusters[:, :num_points], axis=1)
    clusters[:, dist_col_id] = np.amin(clusters[:, :num_points], axis=1)
    
    return X_data, clusters, dist_col_id, clust_col_id


# In[718]:


def complete_link_merge(c1_ind, c2_ind, X_data, clusters, dist_col_id, clust_col_id):
    num_points = clusters.shape[0]

    if c1_ind>c2_ind:
        temp=c1_ind
        c1_ind=c2_ind
        c2_ind=temp
    
    #print(c1_ind, c2_ind)
    for i in range(num_points):
        if clusters[i, c1_ind] < clusters[i, c2_ind]:
            clusters[i, c1_ind]=clusters[i, c2_ind]
            
    for j in range (num_points):
        if clusters[c1_ind, j] < clusters[c2_ind, j]:
            clusters[c1_ind, j]=clusters[c2_ind, j]
            
    clusters=np.delete(clusters, c2_ind, 1)
    clusters=np.delete(clusters, c2_ind, 0)
    np.fill_diagonal(clusters, np.inf)
    num_points = clusters.shape[0]

    for i in range(len(X_data[:,2])):
        if X_data[i, 2]==c2_ind:
            X_data[i, 2]=c1_ind
        if X_data[i,2]>c2_ind:
            X_data[i,2]=X_data[i,2]-1
        
    dist_col_id-=1
    clust_col_id-=1
    
    clusters[:, clust_col_id] = np.argmin(clusters[:, :num_points], axis=1)
    clusters[:, dist_col_id] = np.amin(clusters[:, :num_points], axis=1)
    
    return X_data, clusters, dist_col_id, clust_col_id


# In[719]:


def average_link_merge(c1_ind, c2_ind, X_data, clusters, dist_col_id, clust_col_id):
    num_points = clusters.shape[0]

    if c1_ind>c2_ind:
        temp=c1_ind
        c1_ind=c2_ind
        c2_ind=temp
    
    #print(c1_ind, c2_ind)
    for i in range(num_points):
        clusters[i, c1_ind]=(clusters[i, c1_ind]+clusters[i, c2_ind])/2
            
    for j in range (num_points):
        clusters[c1_ind, j]=(clusters[c1_ind, j]+clusters[c2_ind, j])/2
            
    clusters=np.delete(clusters, c2_ind, 1)
    clusters=np.delete(clusters, c2_ind, 0)
    np.fill_diagonal(clusters, np.inf)
    num_points = clusters.shape[0]

    for i in range(len(X_data[:,2])):
        if X_data[i, 2]==c2_ind:
            X_data[i, 2]=c1_ind
        if X_data[i,2]>c2_ind:
            X_data[i,2]=X_data[i,2]-1
        
    dist_col_id-=1
    clust_col_id-=1
    
    clusters[:, clust_col_id] = np.argmin(clusters[:, :num_points], axis=1)
    clusters[:, dist_col_id] = np.amin(clusters[:, :num_points], axis=1)
    
    return X_data, clusters, dist_col_id, clust_col_id


# In[720]:


threshold=None
merge_distances = np.zeros(num_points - 1)
for i in range(0, num_points - 3):
    c1_ind, c2_ind, distance=find_clusters_to_merge(clusters, dist_col_id, clust_col_id)
    #print(c1_ind, c2_ind)
    #if threshold is set, we don't merge any further if we reached the desired max distance for merging
    if threshold is not None and distance > threshold:
        break
    merge_distances[i] = distance
    merge_distances[i] = distance
    #X_data, clusters, dist_col_id, clust_col_id= single_link_merge(c1_ind, c2_ind, X_data, clusters, dist_col_id, clust_col_id)
    #X_data, clusters, dist_col_id, clust_col_id= complete_link_merge(c1_ind, c2_ind, X_data, clusters, dist_col_id, clust_col_id)
    X_data, clusters, dist_col_id, clust_col_id= average_link_merge(c1_ind, c2_ind, X_data, clusters, dist_col_id, clust_col_id)
    
    #uncomment when testing
    print("Merging clusters #", c1_ind, c2_ind)
    if i%30 == 0:
         for k, (marker, color) in zip(range(num_points), itertools.product(markers, colormap)):
             plt.scatter(X_data[X_data[:, 2] == k, 0], X_data[X_data[:, 2] == k, 1], color=color, marker=marker, label=k)
         plt.show()


# In[721]:


# todo use the plot below to find the optimal threshold to stop merging clusters
plt.plot(np.arange(0, num_points - 1, 1), merge_distances[:num_points - 1])
plt.title("Merge distances over iterations")
plt.xlabel("Iteration #")
plt.ylabel("Distance")
plt.show()

for k, (marker, color) in zip(range(num_points), itertools.product(markers, colormap)):
    plt.scatter(X_data[X_data[:, 2] == k, 0], X_data[X_data[:, 2] == k, 1], color=color, marker=marker)
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()


# In[438]:


c1_ind, c2_ind, distance=find_clusters_to_merge(clusters, dist_col_id, clust_col_id)
print(c1_ind, c2_ind, distance)


# In[439]:


X_data, clusters, dist_col_id, clust_col_id= single_link_merge(c1_ind, c2_ind, X_data, clusters, dist_col_id, clust_col_id)
print(X_data)


# In[514]:


X_data


# In[295]:


np.unique(X_data[:,2], return_counts=True)


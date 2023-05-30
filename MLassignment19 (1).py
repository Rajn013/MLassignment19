#!/usr/bin/env python
# coding: utf-8

# 1. A set of one-dimensional data points is given to you: 5, 10, 15, 20, 25, 30, 35. Assume that k = 2 and that the first set of random centroid is 15, 32, and that the second set is 12, 30.
# a) Using the k-means method, create two clusters for each set of centroid described above.
# b) For each set of centroid values, calculate the SSE.
# 

# In[1]:


import numpy as np
from sklearn.cluster import KMeans

data = np.array([5, 10, 15, 20, 25, 30, 35]).reshape(-1, 1)

set1 = np.array([15, 32]).reshape(-1, 1)
set2 = np.array([12, 30]).reshape(-1, 1)

kmeans_set1 = KMeans(n_clusters=2, random_state=0)
kmeans_set1.fit(data)
clusters_set1 = kmeans_set1.labels_

kmeans_set2 = KMeans(n_clusters=2, random_state=0)
kmeans_set2.fit(data)
clusters_set2 = kmeans_set2.labels_

print("Clusters for centroids_set1:", clusters_set1)
print("Clusters for centroids_set2:", clusters_set2)


# In[12]:


import numpy as np

data = np.array([5, 10, 15, 20, 25, 30, 35])
centroids_set1 = np.array([15, 32])
centroids_set2 = np.array([12, 30])

def calculate_sse(data, centroids):
    sse = 0
    for point in data:
        distances = [(point - centroid) ** 2 for centroid in centroids]
        min_distance = min(distances)
        sse += min_distance
    return sse

sse_set1 = calculate_sse(data_points, centroids_set1)
print(sse_set1)

sse_set2 = calculate_sse(data_points, centroids_set2)
print(sse_set2)


# 2. Describe how the Market Basket Research makes use of association analysis concepts.
# 

# Data preparation: Transactional data is gathered and preprocessed.
# 
# Generating frequent itemsets: Frequent itemsets, sets of items that occur together above a minimum support threshold, are identified using algorithms like Apriori.
# 
# Rule generation: Association rules are generated based on frequent itemsets, with antecedents (items on the left) and consequents (items on the right).
# 
# Rule evaluation: The generated rules are evaluated using metrics like support, confidence, lift, and conviction to measure the strength and significance of the associations.
# 
# Interpretation and action: The rules are interpreted to gain insights and can be used to drive business decisions, such as cross-selling, product placement, recommendations, or targeted marketing campaigns.

# 3. Give an example of the Apriori algorithm for learning association rules.
# 

# In[ ]:


from mlxtend.frequent_patterns import apriori, association_rules
transactions = [
    ['apple', 'banana', 'chocolate'],
    ['banana', 'orange'],
    ['apple', 'banana', 'orange'],
    ['banana', 'chocolate'],
    ['apple', 'chocolate'],
    ['apple', 'banana', 'chocolate', 'orange'],
    ['apple']
]

frequent_itemsets = apriori(transactions, min_support=0.3, use_colnames=True)

rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)

print(rules)


# 4. In hierarchical clustering, how is the distance between clusters measured? Explain how this metric is used to decide when to end the iteration.
# 

# In hierarchical clustering, the distance between clusters is measured using metrics like Euclidean distance or Manhattan distance. The specific metric depends on the data. To decide when to end the iteration, a linkage criterion is used, such as single linkage, complete linkage, average linkage, or Ward's linkage. The iteration stops when a predefined number of clusters is reached, a distance threshold is exceeded, or based on visual analysis of the dendrogram. In Python, you can use the scipy library to perform hierarchical clustering and visualize the dendrogram.

# 5. In the k-means algorithm, how do you recompute the cluster centroids?
# 

# Initialize centroids: Randomly select K data points as the initial centroids.
# 
# Assign data points: Calculate the distance between each data point and the centroids, and assign each data point to the nearest centroid's cluster.
# 
# Recompute centroids: Calculate the mean position of all data points in each cluster, and update the centroids to these new positions.
# 
# Repeat steps 2 and 3 until convergence: Iterate the assignment and centroid update steps until the centroids no longer change significantly or a maximum number of iterations is reached.

# 6. At the start of the clustering exercise, discuss one method for determining the required number of clusters.
# 

# n Clustering algorithms like K-Means clustering, we have to determine the right number of clusters for our dataset. This ensures that the data is properly and efficiently divided. An appropriate value of ‘k’ i.e. the number of clusters helps in ensuring proper granularity of clusters and helps in maintaining a good balance between compressibility and accuracy of clusters.
# 
# Case 1: Treat the entire dataset as one cluster
# Case 2: Treat each data point as a cluster
#     
#     This method is based on the observation that increasing the number of clusters can help in reducing the sum of the within-cluster variance of each cluster. Having more clusters allows one to extract finer groups of data objects that are more similar to each other. For choosing the ‘right’ number of clusters, the turning point of the curve of the sum of within-cluster variances with respect to the number of clusters is used. The first turning point of the curve suggests the right value of ‘k’ for any k > 0. Let us implement the elbow method in Pytho
#     
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans

# 7. Discuss the k-means algorithm's advantages and disadvantages.
# 

# the k-means algoirthm's advantages and disadvantages:
# 
# advantages:
#     1.Relatively simple to implement
#     2.scales to large data sets.
#     3.Guarantees convergence
#     4.can warm-start the position of centroids.
#     5.easy adapts to new example.
#     
# disadvantages:
#     1.choosing k manually.
#     2.Being dependent on initial value.
#     3.clustering data of varying size and density
#     4.clustering outliers.
#     5.scalling with number of dimensions.

# 8. Draw a diagram to demonstrate the principle of clustering.
# 

# In[48]:


import matplotlib.pyplot as plt

data = [ (2, 3), (3, 5), (2, 4),
    (5, 3), (6, 2), (7, 4),
    (9, 7), (10, 8), (8, 9)]

centroids = [(3, 4), (7, 6), (9, 8)]

colors =['red', 'blue','black']

for i, ppoint in enumerate(data):
    plt.scatter(data[i][0], data[i][1], color=colors[i // 3])
    
for centriod in centroids:
    plt.scatter(centroids[0], centroids[1], color="green", marker='x')
    
    
plt.title('10')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()


#  During your study, you discovered seven findings, which are listed in the data points below. Using the K-means algorithm, you want to build three clusters from these observations. The clusters C1, C2, and C3 have the following findings after the first iteration:
# 
# C1: (2,2), (4,4), (6,6); C2: (2,2), (4,4), (6,6); C3: (2,2), (4,4),
# 
# C2: (0,4), (4,0), (0,4), (0,4), (0,4), (0,4), (0,4), (0,4), (0,
# 
# C3: (5,5) and (9,9)
# 

# In[59]:


import numpy as np 
from sklearn.cluster import KMeans

data = np.array([[2, 2], [4, 4], [6, 6],[2, 2], [4, 4], [6, 6],[2, 2], [4, 4],
    [0, 4], [4, 0], [0, 4], [0, 4], [0, 4], [0, 4], [0, 4], [0, 4], [0, 4],
    [5, 5], [9, 9]])
k =  3
kmeans = KMeans(n_clusters=k , random_state= 40)
kmeans.fit(data)

labels = kmeans.labels_
clusters_centers=kmeans.cluster_centers_

for i in range (k):
    indices = np.where(labels ==i )[0]
    findings = data[indices]
    print(f'c{i+1}: {findings}')


#  In a software project, the team is attempting to determine if software flaws discovered during testing are identical. Based on the text analytics of the defect details, they decided to build 5 clusters of related defects. Any new defect formed after the 5 clusters of defects have been identified must be listed as one of the forms identified by clustering. A simple diagram can be used to explain this process. Assume you have 20 defect data points that are clustered into 5 clusters and you used the k-means algorithm.
# 

# In[62]:


import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


np.random.seed(42)
defect_data_points = np.random.rand(20, 2) * 10


k = 5
kmeans = KMeans(n_clusters=k, random_state=42).fit(defect_data_points)


plt.scatter(defect_data_points[:, 0], defect_data_points[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', marker='x')

plt.title('rajn')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.show()


# In[ ]:





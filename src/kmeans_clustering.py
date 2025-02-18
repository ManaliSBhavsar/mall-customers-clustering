import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/Mall_Customers.csv')
print('REVIEW OF MALL CUSTOMERS DATASET:')
print(df.head(10))
x = df.iloc[:, [3,4]].values

#GENERATING TRAINING AND TESTING DATASET
X_train, X_test = train_test_split(x, test_size = 0.5, random_state = 0)

#ELBOW METHOD TO FIND K
print('\nELBOW METHOD:')
Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)
plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()
print('From the graph, it is clear that the elbow shape is formed at value 5. Hence, the optimized value of K=5.')

#KMEANS ON TRAINING DATASET
print('\n*************** TRAINING DATASET ***************')
kmeans = KMeans(n_clusters=5)
y_kmeans = kmeans.fit_predict(X_train)
#print(y_kmeans5)

print('\nCENTROIDS CALCULATED FOR K CLUSTERS :')
print(kmeans.cluster_centers_)

print('\nVISUAL REPRESENTATION BEFORE CLUSTERING:')
plt.scatter(X_train[:,0],X_train[:,1],s=10,c='black',label='Data point')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show() 

print('\nVISUAL REPRESENTATION OF CLUSTERS FORMED:')
plt.scatter(X_train[:,0],X_train[:,1],s=10,c=y_kmeans,cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=30,c='black',label='Centroid')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show() 

#KMEANS ON TESTING DATASET
print('\n*************** TESTING DATASET ***************')
kmeans = KMeans(n_clusters=5)
y_kmeans = kmeans.fit_predict(X_test)
#print(y_kmeans5)

print('\nCENTROIDS CALCULATED FOR K CLUSTERS :')
print(kmeans.cluster_centers_)

print('\nVISUAL REPRESENTATION BEFORE CLUSTERING:')
plt.scatter(X_test[:,0],X_test[:,1],s=10,c='black',label='Data point')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show() 

print('\nVISUAL REPRESENTATION OF CLUSTERS FORMED:')
plt.scatter(X_test[:,0],X_test[:,1],s=10,c=y_kmeans,cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=30,c='black',label='Centroid')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show() 

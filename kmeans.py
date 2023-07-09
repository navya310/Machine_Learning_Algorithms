

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Load the Mall Customer dataset
data = pd.read_csv('/content/drive/MyDrive/machine_learning/Mall_Customers.csv')  # Replace 'mall_customer_dataset.csv' with the actual path to your dataset

# Step 2: Select features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 3: Perform k-means clustering
k = 5  # Set the desired number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Step 4: Get cluster labels
labels = kmeans.labels_

# Step 5: Plot the clusters
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=labels)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, marker='X', c='red')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('K-means Clustering on Mall Customer Dataset')
plt.show()
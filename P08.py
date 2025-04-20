#code to extract Iris.csv data set form inbuid file
import pandas as pd
from sklearn.datasets import load_iris

# Load the iris dataset
iris_data = load_iris()

# Create DataFrame
df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)

# Add the target variable (species) and map the target to actual names
df['target'] = iris_data.target
df['species'] = df['target'].apply(lambda x: iris_data.target_names[x])

# Save to CSV
df.to_csv('/home/shan/Desktop/DSA/Iris.csv', index=False)

print("Iris.csv has been created successfully!")


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.cluster as cluster
from sklearn.cluster import KMeans
import seaborn as sns 
import sklearn.metrics as metrics

# Load dataset
dataset = pd.read_csv('/home/shan/Desktop/DSA/Iris.csv')
x = dataset.iloc[:, [0, 1, 2, 3]].values
print(x)

# Elbow method to find optimal clusters
K = range(1, 10)
wss = []
for k in K:
    kmeans = cluster.KMeans(n_clusters=k, init="k-means++", random_state=42)
    kmeans.fit(x)
    wss.append(kmeans.inertia_)

# Create DataFrame for Elbow plot
elbow_df = pd.DataFrame({'Clusters': list(K), 'WSS': wss})
print(elbow_df)

# Plot Elbow graph
sns.lineplot(x='Clusters', y='WSS', data=elbow_df, marker="o")
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters")
plt.ylabel("Within-Cluster Sum of Square")
plt.show()

# Silhouette Analysis
SK = range(3, 10)
sil_score = []

for i in SK:
    labels = KMeans(n_clusters=i, init="k-means++", random_state=100).fit(x).labels_
    score = metrics.silhouette_score(x, labels, metric="euclidean", sample_size=150, random_state=100)
    sil_score.append(score)
    print(f"Silhouette score for k={i} is {score:.4f}")

# Silhouette DataFrame
sil_centers = pd.DataFrame({'Clusters': list(SK), 'Sil Score': sil_score})
print(sil_centers)

# Perform KMeans with 3 clusters (based on Elbow/Silhouette)
kmeans = KMeans(n_clusters=3, init="k-means++", random_state=42)
y_kmeans = kmeans.fit_predict(x)

# Visualize clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], c='red', label='Setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], c='blue', label='Versicolor')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], c='green', label='Virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='yellow', label='Centroids')
plt.title("K-Means Clustering (k=3)")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.show()

# Silhouette Lineplot
sns.lineplot(x='Clusters', y='Sil Score', data=sil_centers, marker='o')
plt.title("Silhouette Score for Different k")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()

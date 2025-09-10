import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Load the dataset
data = pd.read_csv('/mnt/data/Mall_Customers.csv')

# 2. Select features for clustering
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

# 3. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Elbow Method to choose k
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.show()

# 5. Train KMeans model (choose k after elbow, e.g. k=5)
k_optimal = 5
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# 6. Visualize the clusters
plt.figure(figsize=(8, 6))
for cluster in range(k_optimal):
    plt.scatter(
        X_scaled[data['Cluster'] == cluster, 0],
        X_scaled[data['Cluster'] == cluster, 1],
        label=f'Cluster {cluster}'
    )

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200, c='black', marker='X', label='Centroids'
)

plt.title('Customer Segments (K-Means)')
plt.xlabel(features[0] + ' (scaled)')
plt.ylabel(features[1] + ' (scaled)')
plt.legend()
plt.show()

# 7. Save clustered data
data.to_csv('clustered_customers.csv', index=False)
print("Clusters added to dataset and saved as clustered_customers.csv")

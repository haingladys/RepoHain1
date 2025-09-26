import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

file_path = "/AmazonSaleReport.csv"
data = pd.read_csv(file_path)

data = data.dropna()
print(data.head())

features = ['Qty', 'Amount']

data = data.dropna(subset=['Amount'])

df = data[features]

# Handle outliers using Z-score
z_scores = df.apply(zscore)
df = df[(z_scores < 3).all(axis=1)]

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

## Clustering

from sklearn.cluster import KMeans

# Apply K-Means clustering
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(data[['Qty', 'Amount']])

data['Cluster'] = clusters

## Feature Engineering (Placeholder)

def feature_engineering(data):
    """Performs feature engineering if needed."""
    return data

data = feature_engineering(data)

## Visualization

import matplotlib.pyplot as plt
import seaborn as sns

# Elbow Method to find optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k')
plt.show()

# Visualizing Customer Segments
plt.figure(figsize=(8,5))
sns.scatterplot(x=data['Qty'], y=data['Amount'],
                hue=data['Cluster'], palette='viridis', s=100)
plt.xlabel('Quantity Purchased')
plt.ylabel('Total Amount Spent')
plt.title('Customer Segments')
plt.legend()
plt.show()

# Save clustered data
data.to_csv("/content/sample_data.csv", index=False) # Changed file path to include '.csv' extension

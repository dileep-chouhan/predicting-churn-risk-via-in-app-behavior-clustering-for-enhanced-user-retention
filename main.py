import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_users = 500
data = {
    'Daily_Logins': np.random.randint(0, 5, size=num_users),
    'Avg_Session_Duration': np.random.randint(1, 60, size=num_users),
    'Features_Used': np.random.randint(1, 10, size=num_users),
    'Purchases': np.random.randint(0, 3, size=num_users),
    'Churn': np.random.choice([0, 1], size=num_users, p=[0.8, 0.2]) # 20% churn rate
}
df = pd.DataFrame(data)
# --- 2. Data Preprocessing ---
# Scale the data for clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.drop('Churn', axis=1))
# --- 3. Clustering ---
# Determine optimal number of clusters (e.g., using the Elbow method - simplified here)
kmeans = KMeans(n_clusters=3, random_state=42) # Choosing 3 clusters as an example.  A more robust method would be needed in a real-world scenario.
df['Cluster'] = kmeans.fit_predict(scaled_data)
# --- 4. Dimensionality Reduction (for visualization) ---
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
df['PCA1'] = pca_data[:, 0]
df['PCA2'] = pca_data[:, 1]
# --- 5. Visualization ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='viridis')
plt.title('User Clusters based on In-App Behavior')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.tight_layout()
# Save the cluster plot
output_filename_cluster = 'user_clusters.png'
plt.savefig(output_filename_cluster)
print(f"Plot saved to {output_filename_cluster}")
# --- 6. Churn Analysis per Cluster ---
churn_by_cluster = df.groupby('Cluster')['Churn'].mean()
print("\nChurn Rate per Cluster:")
print(churn_by_cluster)
# --- 7. Visualization of Churn Rate per Cluster (Bar Plot) ---
plt.figure(figsize=(8, 6))
sns.barplot(x=churn_by_cluster.index, y=churn_by_cluster.values)
plt.title('Churn Rate by User Cluster')
plt.xlabel('Cluster')
plt.ylabel('Churn Rate')
plt.grid(True)
plt.tight_layout()
# Save the churn rate plot
output_filename_churn = 'churn_rate_by_cluster.png'
plt.savefig(output_filename_churn)
print(f"Plot saved to {output_filename_churn}")
print("\nAnalysis Complete.  Note: This uses a simplified approach to cluster selection and churn prediction.  A more robust model would be necessary for a production environment.")
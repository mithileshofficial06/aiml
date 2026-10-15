# Experiment 6: K-Means Clustering with the Elbow Method
#
# Algorithm:
# 1. Import the required libraries and load the Iris dataset (built into sklearn).
# 2. Select two features (petal length and petal width) for clustering.
# 3. Run K-Means for K = 1 to 10 and store the WCSS (inertia) for each K.
# 4. Plot the Elbow graph and choose the K where the curve bends (K = 3).
# 5. Train K-Means with K = 3 and assign a cluster to every data point.
# 6. Plot the clusters and their centroids.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# 1. Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print("Sample Data:")
print(df.head())

# 2. Select features
X = df[["petal length (cm)", "petal width (cm)"]]

# 3. Find WCSS for K = 1 to 10
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    wcss.append(km.inertia_)

# 4. Elbow graph
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.show()

# 5. Train K-Means with K = 3
model = KMeans(n_clusters=3, n_init=10, random_state=42)
df["Cluster"] = model.fit_predict(X)

print("\nClustered Data:")
print(df.head())
print("\nCluster Centers:")
print(model.cluster_centers_)

# 6. Plot clusters and centroids
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df["Cluster"], s=30)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
            c="red", marker="X", s=200, label="Centroids")
plt.title("K-Means Clustering")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()
plt.show()

# ============================================================
# OUTPUT
# ============================================================
# Sample Data:
#    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# 0                5.1               3.5                1.4               0.2
# 1                4.9               3.0                1.4               0.2
# 2                4.7               3.2                1.3               0.2
# 3                4.6               3.1                1.5               0.2
# 4                5.0               3.6                1.4               0.2
#
# Clustered Data:
#    sepal length (cm)  sepal width (cm)  ...  petal width (cm)  Cluster
# 0                5.1               3.5  ...               0.2        1
# 1                4.9               3.0  ...               0.2        1
# 2                4.7               3.2  ...               0.2        1
# 3                4.6               3.1  ...               0.2        1
# 4                5.0               3.6  ...               0.2        1
#
# [5 rows x 5 columns]
#
# Cluster Centers:
# [[5.59583333 2.0375    ]
#  [1.462      0.246     ]
#  [4.26923077 1.34230769]]
#
# (Graph output: see outputs/exp6_kmeans_elbow_plot1.png, outputs/exp6_kmeans_elbow_plot2.png)

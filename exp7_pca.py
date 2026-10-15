# Experiment 7: Principal Component Analysis (PCA)
#
# Algorithm:
# 1. Import the required libraries and load the Iris dataset (4 features).
# 2. Standardize the features so that each has mean 0 and variance 1.
# 3. Apply PCA to reduce the 4 features to 2 principal components.
# 4. Display the explained variance ratio of each component.
# 5. Display the data before and after reduction.
# 6. Plot the data on the two principal components.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print("Original Data (4 features):")
print(X.head())
print("Shape:", X.shape)

# 2. Standardize
X_scaled = StandardScaler().fit_transform(X)

# 3. Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Explained variance
print("\nExplained Variance Ratio:", pca.explained_variance_ratio_.round(3))
print("Total Variance Kept:", round(sum(pca.explained_variance_ratio_) * 100, 2), "%")

# 5. Reduced data
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"]).round(3)
print("\nData after PCA (2 features):")
print(df_pca.head())
print("Shape:", df_pca.shape)

# 6. Plot
for label, name in enumerate(iris.target_names):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=name, s=30)

plt.title("PCA of Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()

# ============================================================
# OUTPUT
# ============================================================
# Original Data (4 features):
#    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# 0                5.1               3.5                1.4               0.2
# 1                4.9               3.0                1.4               0.2
# 2                4.7               3.2                1.3               0.2
# 3                4.6               3.1                1.5               0.2
# 4                5.0               3.6                1.4               0.2
# Shape: (150, 4)
#
# Explained Variance Ratio: [0.73  0.229]
# Total Variance Kept: 95.81 %
#
# Data after PCA (2 features):
#      PC1    PC2
# 0 -2.265  0.480
# 1 -2.081 -0.674
# 2 -2.364 -0.342
# 3 -2.299 -0.597
# 4 -2.390  0.647
# Shape: (150, 2)
#
# (Graph output: see outputs/exp7_pca_plot1.png)

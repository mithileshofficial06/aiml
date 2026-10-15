# Experiment 5: Support Vector Machine (SVM)
#
# Algorithm:
# 1. Import the required libraries and load winequality-red.csv.
# 2. Select the input features (X) and the target "quality" (y).
# 3. Split the dataset into training and testing data.
# 4. Scale the features and train the SVM model.
# 5. Predict the test data and display the confusion matrix and classification report.
# 6. Plot an SVM decision boundary with support vectors using sample 2D data.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.datasets import make_blobs

# Connect Google Drive (to read the CSV file)
from google.colab import drive
drive.mount("/content/drive")

# 1. Load dataset
df = pd.read_csv("/content/drive/MyDrive/AIML LAB/winequality-red.csv")
print("Sample Data:")
print(df.head())

# 2. Features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale the features and train SVM
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC()
model.fit(X_train, y_train)

# 5. Predict and evaluate
y_pred = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Actual vs Predicted:")
print(pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).head(10))

# 6. Plot SVM decision boundary (sample 2D data)
X2, y2 = make_blobs(n_samples=100, centers=2, random_state=0)
svm = SVC(kernel="linear").fit(X2, y2)

plt.scatter(X2[:, 0], X2[:, 1], c=y2)
plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
            s=100, facecolors="none", edgecolors="red", label="Support Vectors")

xx, yy = np.meshgrid(np.linspace(*plt.xlim(), 100), np.linspace(*plt.ylim(), 100))
Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"])

plt.title("SVM Decision Boundary")
plt.legend()
plt.show()

# ============================================================
# OUTPUT
# ============================================================
# Mounted at /content/drive
# Sample Data:
#    fixed acidity  volatile acidity  citric acid  ...  sulphates  alcohol  quality
# 0            7.4              0.70         0.00  ...       0.56      9.4        5
# 1            7.8              0.88         0.00  ...       0.68      9.8        5
# 2            7.8              0.76         0.04  ...       0.65      9.8        5
# 3           11.2              0.28         0.56  ...       0.58      9.8        6
# 4            7.4              0.70         0.00  ...       0.56      9.4        5
#
# [5 rows x 12 columns]
#
# Accuracy: 60.31 %
#
# Confusion Matrix:
# [[ 0  0  1  0  0  0]
#  [ 0  0  8  2  0  0]
#  [ 0  0 99 31  0  0]
#  [ 0  0 43 85  4  0]
#  [ 0  0  1 32  9  0]
#  [ 0  0  0  2  3  0]]
#
# Classification Report:
#               precision    recall  f1-score   support
#
#            3       0.00      0.00      0.00         1
#            4       0.00      0.00      0.00        10
#            5       0.65      0.76      0.70       130
#            6       0.56      0.64      0.60       132
#            7       0.56      0.21      0.31        42
#            8       0.00      0.00      0.00         5
#
#     accuracy                           0.60       320
#    macro avg       0.30      0.27      0.27       320
# weighted avg       0.57      0.60      0.57       320
#
# Actual vs Predicted:
#    Actual  Predicted
# 0       6          5
# 1       5          5
# 2       6          6
# 3       5          5
# 4       6          6
# 5       5          5
# 6       5          5
# 7       5          5
# 8       5          6
# 9       6          6
#
# (Graph output: see outputs/exp5_svm_plot1.png)

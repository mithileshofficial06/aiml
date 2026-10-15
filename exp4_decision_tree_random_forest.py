# Experiment 4: Decision Tree and Random Forest (PlayTennis dataset)
#
# Algorithm:
# 1. Import the required libraries and load PlayTennis.csv.
# 2. Convert text values (Sunny, Hot, Yes...) into numbers using Label Encoding.
# 3. Split the dataset into training and testing data.
# 4. Train a Decision Tree (entropy) and predict the test data.
# 5. Train a Random Forest (many decision trees, majority vote) and predict the test data.
# 6. Compare the accuracy of both models and draw the decision tree.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Connect Google Drive (to read the CSV file)
from google.colab import drive
drive.mount("/content/drive")

# 1. Load dataset
df = pd.read_csv("/content/drive/MyDrive/AIML LAB/PlayTennis.csv")
print("Sample Data:")
print(df.head())

# 2. Label Encoding
for col in df.columns:
    df[col] = LabelEncoder().fit_transform(df[col])

print("\nPreprocessed Data:")
print(df.head())

# 3. Split features and target
X = df.drop("play", axis=1)
y = df["play"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Decision Tree
dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# 5. Random Forest
rf = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# 6. Compare results
print("\nActual Values:          ", list(y_test))
print("Decision Tree Predicted:", dt_pred.tolist())
print("Random Forest Predicted:", rf_pred.tolist())
print("\nDecision Tree Accuracy:", round(accuracy_score(y_test, dt_pred) * 100, 2), "%")
print("Random Forest Accuracy:", round(accuracy_score(y_test, rf_pred) * 100, 2), "%")

plt.figure(figsize=(10, 6))
plot_tree(dt, feature_names=list(X.columns), class_names=["No", "Yes"], filled=True)
plt.title("Decision Tree")
plt.show()

# ============================================================
# OUTPUT
# ============================================================
# Mounted at /content/drive
# Sample Data:
#     outlook  temp humidity   windy play
# 0     Sunny   Hot     High    Weak   No
# 1     Sunny   Hot     High  Strong   No
# 2  Overcast   Hot     High    Weak  Yes
# 3      Rain  Mild     High    Weak  Yes
# 4      Rain  Cool   Normal    Weak  Yes
#
# Preprocessed Data:
#    outlook  temp  humidity  windy  play
# 0        2     1         0      1     0
# 1        2     1         0      0     0
# 2        0     1         0      1     1
# 3        1     2         0      1     1
# 4        1     0         1      1     1
#
# Actual Values:           [1, 1, 0, 1, 0]
# Decision Tree Predicted: [1, 0, 0, 1, 1]
# Random Forest Predicted: [1, 0, 0, 1, 1]
#
# Decision Tree Accuracy: 60.0 %
# Random Forest Accuracy: 60.0 %
#
# (Graph output: see outputs/exp4_decision_tree_random_forest_plot1.png)

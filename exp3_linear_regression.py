# Experiment 3: Simple Linear Regression and Multiple Linear Regression
#
# Algorithm:
# 1. Import the required libraries and load housing.csv; remove missing values.
# 2. Select median_house_value as the target (y).
# 3. Simple Linear Regression: use only median_income as input, split, train, predict.
# 4. Multiple Linear Regression: use all numeric columns as input, split, train, predict.
# 5. Display actual vs predicted values and the R2 score of both models.
# 6. Plot graphs to compare actual and predicted house values.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Connect Google Drive (to read the CSV file)
from google.colab import drive
drive.mount("/content/drive")

# 1. Load dataset
df = pd.read_csv("/content/drive/MyDrive/AIML LAB/housing.csv")
df = df.dropna()

print("Sample Data:")
print(df.head())

# 2. Target
y = df["median_house_value"]

# 3. Simple Linear Regression (one input)
X1 = df[["median_income"]]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.2, random_state=42)

model1 = LinearRegression()
model1.fit(X1_train, y1_train)
pred1 = model1.predict(X1_test)

# 4. Multiple Linear Regression (many inputs)
X2 = df.select_dtypes(include="number").drop(columns=["median_house_value"])
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.2, random_state=42)

model2 = LinearRegression()
model2.fit(X2_train, y2_train)
pred2 = model2.predict(X2_test)

# 5. Results
print("\nSimple Regression R2 Score  :", round(r2_score(y1_test, pred1), 2))
print("Multiple Regression R2 Score:", round(r2_score(y2_test, pred2), 2))

print("\nActual vs Predicted (Multiple Regression):")
print(pd.DataFrame({"Actual": y2_test[:10].values, "Predicted": pred2[:10].round(0)}))

# 6. Plot graphs
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X1_test, y1_test, s=10, label="Actual")
plt.scatter(X1_test, pred1, s=10, label="Predicted")
plt.title("Simple Linear Regression")
plt.xlabel("Median Income")
plt.ylabel("House Value")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y2_test, pred2, s=10)
plt.plot([y2_test.min(), y2_test.max()], [y2_test.min(), y2_test.max()], "r--")
plt.title("Multiple Linear Regression")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.tight_layout()
plt.show()

# ============================================================
# OUTPUT
# ============================================================
# Mounted at /content/drive
# Sample Data:
#    longitude  latitude  ...  median_house_value  ocean_proximity
# 0    -122.23     37.88  ...            452600.0         NEAR BAY
# 1    -122.22     37.86  ...            358500.0         NEAR BAY
# 2    -122.24     37.85  ...            352100.0         NEAR BAY
# 3    -122.25     37.85  ...            341300.0         NEAR BAY
# 4    -122.25     37.85  ...            342200.0         NEAR BAY
#
# [5 rows x 10 columns]
#
# Simple Regression R2 Score  : 0.47
# Multiple Regression R2 Score: 0.64
#
# Actual vs Predicted (Multiple Regression):
#      Actual  Predicted
# 0  245800.0   197058.0
# 1  137900.0   157509.0
# 2  218200.0   202099.0
# 3  220800.0   173502.0
# 4  170500.0   213795.0
# 5   75000.0   108005.0
# 6  269400.0   284796.0
# 7  228900.0   249129.0
# 8  500001.0   391339.0
# 9   80800.0    49301.0
#
# (Graph output: see outputs/exp3_linear_regression_plot1.png)

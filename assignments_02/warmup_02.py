# --- scikit-learn API ---

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


os.makedirs("outputs", exist_ok=True)


# scikit-learn API Q1

years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

linear_model1 = LinearRegression()
linear_model1.fit(years, salary)

slope = linear_model1.coef_[0]
intercept = linear_model1.intercept_

prediction_4 = linear_model1.predict([[4]])[0]
prediction_8 = linear_model1.predict([[8]])[0]

print("Slope:", slope)
print("Intercept:", intercept)
print("Prediction for 4 years:", prediction_4)
print("Prediction for 8 years:", prediction_8)


# scikit-learn API Q2

x = np.array([10, 20, 30, 40, 50])

print("\nOriginal shape:", x.shape)

x = x.reshape(-1, 1)

print("2D shape:", x.shape)

# scikit-learn expects X to be 2D because the rows represent samples
# and the columns represent features.
#
# Here we have 5 samples and 1 feature, so the required shape is (5, 1).
# Keeping X as shape (5,) would not tell scikit-learn how many features
# each sample contains.


# scikit-learn API Q3

X_clusters, _ = make_blobs(
    n_samples=120,
    centers=3,
    cluster_std=0.8,
    random_state=7
)

kmeans_model1 = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans_model1.fit(X_clusters)

centers = kmeans_model1.cluster_centers_
labels = kmeans_model1.labels_

print("\nCluster Centers:")
print(centers)

print("\nPoints in Each Cluster:")
print(np.bincount(labels))


plt.figure(figsize=(8, 6))

plt.scatter(
    X_clusters[:, 0],
    X_clusters[:, 1],
    c=labels,
    cmap="viridis"
)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c="black",
    s=200,
    marker="X",
    label="Cluster Centers"
)

plt.title("K-Means Clustering of Synthetic Data into 3 Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.savefig(os.path.join("outputs", "kmeans_clusters.png"))
plt.close()


# --- Linear Regression ---

np.random.seed(42)

num_patients = 100

age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)

cost = (
    200 * age
    + 15000 * smoker
    + np.random.normal(0, 3000, num_patients)
)


# Linear Regression Q1

plt.figure(figsize=(8, 6))

plt.scatter(
    age,
    cost,
    c=smoker,
    cmap="coolwarm"
)

plt.title("Medical Cost vs Patient Age")
plt.xlabel("Patient Age")
plt.ylabel("Medical Cost")

plt.savefig(os.path.join("outputs", "cost_vs_age.png"))
plt.close()


# Linear Regression Q2

X = age.reshape(-1, 1)
y = cost

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Linear Regression Q3

model = LinearRegression()

model.fit(X_train, y_train)

print("\nAge Only Model")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

y_pred = model.predict(X_test)

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)

print("RMSE:", rmse)
print("R²:", r2)

# The positive coefficient means the model predicts medical cost will
# increase as age increases.
#
# The age-only model is also missing an important variable: smoking.
# Looking at the graph, smokers have much higher medical costs even at
# similar ages, so age by itself cannot explain all of the differences.


# Linear Regression Q4

X_full = np.column_stack([age, smoker])

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full,
    cost,
    test_size=0.20,
    random_state=42
)

model_full = LinearRegression()

model_full.fit(X_train_full, y_train_full)

r2_full = model_full.score(X_test_full, y_test_full)

print("\nAge + Smoker Model")
print("R² (Age Only Model):", r2)
print("R² (Age + Smoker Model):", r2_full)

print("Age Coefficient:", model_full.coef_[0])
print("Smoker Coefficient:", model_full.coef_[1])

# Adding the smoker feature increases R² because smoking is an important
# part of how the data was generated.
#
# The age coefficient represents the predicted increase in medical cost
# for one additional year of age while smoking status stays the same.
#
# The smoker coefficient represents how much higher the predicted cost is
# for a smoker compared with a non-smoker of the same age.


# Linear Regression Q5

y_pred_full = model_full.predict(X_test_full)

plt.figure(figsize=(8, 6))

plt.scatter(y_pred_full, y_test_full)

min_val = min(y_pred_full.min(), y_test_full.min())
max_val = max(y_pred_full.max(), y_test_full.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val]
)

plt.title("Predicted vs Actual Medical Costs")
plt.xlabel("Predicted Medical Cost")
plt.ylabel("Actual Medical Cost")

plt.savefig(os.path.join("outputs", "predicted_vs_actual.png"))
plt.close()

# The diagonal represents a perfect prediction where predicted cost equals
# actual cost.
#
# A point above the line means the actual cost was higher than predicted,
# so the model underpredicted.
#
# A point below the line means the actual cost was lower than predicted,
# so the model overpredicted.

# --- scikit-learn API ---

# scikit-learn API Q1
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

linear_model1 = LinearRegression()
linear_model1.fit(years, salary)

m = linear_model1.coef_[0]
b = linear_model1.intercept_

years4 = m*4 + b
years8 = m*8 + b

print(f"Person with 4 years: {years4}")
print(f"Person with 8 years: {years8}")
print(f"Coef: {m}")
print(f"Y-intercept: {b}")

# scikit-learn API Q2
x = np.array([10, 20, 30, 40, 50])
print(f"Original shape: {x.shape}")

x = x.reshape(-1,1)
print(f"After shape: {x.shape}")



# The function is built to handle multi-linear such as having multiple features so easier to have the assumed to be samples and specify the amount of features as shown only one column.

# scikit-learn API Q3



X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)
kmeans_model1 = KMeans(n_clusters=3, random_state=42)
kmeans_model1.fit(X_clusters)
centers = kmeans_model1.cluster_centers_
labels = kmeans_model1.labels_
print(f"Cluster Labels: {centers}")
print(f"Points in cluster: {np.bincount(labels)}")

plt.scatter(
    X_clusters[:, 0],
    X_clusters[:, 1],
    c=labels,
    cmap='viridis'
)
plt.title("kmeans_clusters")
plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c='black',
    s=200,
    marker='X'
)
os.makedirs("outputs",exist_ok = True)
plt.savefig(os.path.join("outputs", "kmeans_clusters.png"))
plt.show()



plt.plot(labels)

# --- Linear Regression ---

np.random.seed(42)

num_patients = 100

age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)

cost = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

os.makedirs("outputs", exist_ok=True)

# Linear Regression Q1

plt.figure(figsize=(8,6))
plt.scatter(age, cost, c=smoker, cmap="coolwarm")

plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Medical Cost")

plt.savefig(os.path.join("outputs", "cost_vs_age.png"))
plt.show()


# Linear Regression Q2

X = age.reshape(-1, 1)
y = cost

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)



# Linear Regression Q3

model = LinearRegression()

model.fit(X_train, y_train)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

y_pred = model.predict(X_test)

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)

print("RMSE:", rmse)
print("R²:", r2)

# The coef suggests a fast positive rate of change in the price of healthcare. 
# It could be an effect of the smokers because looking at the graph there is a division
# in smokers medical costs vs non smokers even regardless of age.


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

print("R² (Age only model):", r2)
print("R² (Age + Smoker model):", r2_full)

print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])

# As suggested previously adding the smoker flag did help predictions and increases the r^2
# The smoker coef suggests a higher rate of change in medical costs 

# Linear Regression Q5

y_pred_full = model_full.predict(X_test_full)

plt.figure(figsize=(8,6))

plt.scatter(y_pred_full, y_test_full)

min_val = min(y_pred_full.min(), y_test_full.min())
max_val = max(y_pred_full.max(), y_test_full.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val]
)

plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")

plt.savefig(os.path.join("outputs", "predicted_vs_actual.png"))
plt.show()

# The diagonal would be if the model was correct in it's perdiction. Above is the model overpredicting and below the model is under perdicting




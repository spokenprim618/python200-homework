import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


iris = load_iris()
X = iris["data"]
y = iris["target"]

print("\n=== Preprocessing Q1 ===")
# === Preprocessing Q1 ===

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)

print("\n=== Preprocessing Q2 ===")
# === Preprocessing Q2 ===
# You would only on the test becuse you dont want data leak

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Column means of X_train_scaled:")
print(X_train_scaled.mean(axis=0))

print("\n=== KNN Q1 ===")
# === KNN Q1 ===

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

acc_knn = accuracy_score(y_test, y_pred_knn)

print("Accuracy:", acc_knn)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))

print("\n=== KNN Q2 ===")
# === KNN Q2 ===
#It seems to have boosted the accuracy. It may have boosted the accuracy because now the model finds it easier to find patterns in data that is similiar in scale and reduces variation

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)

y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)

acc_knn_scaled = accuracy_score(y_test, y_pred_knn_scaled)

print("Scaled Accuracy:", acc_knn_scaled)

print("\n=== KNN Q3 ===")
# === KNN Q3 ===
# Due to variation having multple adds to the validity of the results

cv_scores = cross_val_score(
    KNeighborsClassifier(n_neighbors=5),
    X_train,
    y_train,
    cv=5
)

print("Fold scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())
print("Standard deviation:", cv_scores.std())

print("\n=== KNN Q4 ===")
# === KNN Q4 ===

k_values = [1, 3, 5, 7, 9, 11, 13, 15]

best_k = None
best_score = 0

for k in k_values:
    scores = cross_val_score(
        KNeighborsClassifier(n_neighbors=k),
        X_train,
        y_train,
        cv=5
    )

    mean_score = scores.mean()

    print(f"k={k:2d}  Mean CV Accuracy={mean_score:.4f}")

    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print(f"\nSuggested k: {best_k}")

print("\n=== Classifier Evaluation Q1 ===")
# === Classifier Evaluation Q1 ===
# It seems it worked well and didnt confuse

cm = confusion_matrix(y_test, y_pred_knn)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris["target_names"]
)

disp.plot(cmap="Blues")
plt.title("KNN Confusion Matrix")
plt.savefig("outputs/knn_confusion_matrix.png")
plt.close()

print("\n=== Decision Tree Q1 ===")
# === Decision Tree Q1 ===
# The two are pretty similar but knn is slightly better
# It wouldn't make a difference because scaled only imacts distance

tree = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)

tree_acc = accuracy_score(y_test, y_pred_tree)

print("Accuracy:", tree_acc)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tree))

print("\n=== Logistic Regression Q1 ===")
# === Logistic Regression Q1 ===
# It increases as well.
# What is allowed is looser more is coming through
lr_001 = LogisticRegression(
    C=0.01,
    max_iter=1000,
    solver="lbfgs"
)

lr_001.fit(X_train_scaled, y_train)

print("C = 0.01")
print("Total coefficient magnitude:",
      np.abs(lr_001.coef_).sum())
print()


lr_1 = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="lbfgs"
)

lr_1.fit(X_train_scaled, y_train)

print("C = 1.0")
print("Total coefficient magnitude:",
      np.abs(lr_1.coef_).sum())
print()


lr_100 = LogisticRegression(
    C=100,
    max_iter=1000,
    solver="lbfgs"
)

lr_100.fit(X_train_scaled, y_train)

print("C = 100")
print("Total coefficient magnitude:",
      np.abs(lr_100.coef_).sum())
print()

digits = load_digits()

X_digits = digits["data"]
y_digits = digits["target"]
images = digits["images"]

print("\n=== PCA Q1 ===")
# === PCA Q1 ===

print("X_digits shape:", X_digits.shape)
print("images shape :", images.shape)

fig, axes = plt.subplots(1, 10, figsize=(15, 2))

for digit in range(10):
    idx = np.where(y_digits == digit)[0][0]

    axes[digit].imshow(
        images[idx],
        cmap="gray_r"
    )

    axes[digit].set_title(str(digit))
    axes[digit].axis("off")

plt.tight_layout()
plt.savefig("outputs/sample_digits.png")
plt.close()

print("\n=== PCA Q2 ===")
# === PCA Q2 ===
# Yes like 1,2,5, and 8

pca = PCA()
pca.fit(X_digits)

scores = pca.transform(X_digits)

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    scores[:, 0],
    scores[:, 1],
    c=y_digits,
    cmap="tab10",
    s=10
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection of Digits")
plt.colorbar(scatter, label="Digit")

plt.savefig("outputs/pca_2d_projection.png")
plt.close()

print("\n=== PCA Q3 ===")
# === PCA Q3 ===
# Around 12 and 13

cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance
)

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Variance Explained")

plt.grid(True)

plt.savefig("outputs/pca_variance_explained.png")
plt.close()



print("\n=== PCA Q4 ===")
# === PCA Q4 ===
# Around 10 which is where it starts to level off
def reconstruct_digit(sample_idx, scores, pca, n_components):
    reconstruction = pca.mean_.copy()

    for i in range(n_components):
        reconstruction = (
            reconstruction
            + scores[sample_idx, i] * pca.components_[i]
        )

    return reconstruction.reshape(8, 8)


sample_idx = 0

fig, axes = plt.subplots(1, 5, figsize=(12, 3))

axes[0].imshow(images[sample_idx], cmap="gray_r")
axes[0].set_title("Original")
axes[0].axis("off")

for ax, n in zip(axes[1:], [2, 5, 10, 20]):
    reconstruction = reconstruct_digit(
        sample_idx,
        scores,
        pca,
        n
    )

    ax.imshow(reconstruction, cmap="gray_r")
    ax.set_title(f"{n} PCs")
    ax.axis("off")

plt.tight_layout()
plt.savefig("outputs/pca_reconstructions.png")
plt.close()

print("Saved PCA reconstruction figure.")
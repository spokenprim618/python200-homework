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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


os.makedirs("outputs", exist_ok=True)


# --- Required Iris Setup ---

iris = load_iris(as_frame=True)

X = iris["data"]
y = iris["target"]

# --- Preprocessing Q1 ---

print("\n=== Preprocessing Q1 ===")

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


# --- Preprocessing Q2 ---

print("\n=== Preprocessing Q2 ===")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)


X_test_scaled = scaler.transform(X_test) 
print("Column means of X_train_scaled:")
print(X_train_scaled.mean(axis=0))

# StandardScaler is fit only on X_train so the mean and standard
# deviation are learned without using any information from X_test.
# X_test is then transformed using those same training statistics.
#
# The printed means of X_train_scaled should all be very close to 0,
# which confirms that the training features were centered by the scaler.


# --- KNN Q1 ---

print("\n=== KNN Q1 ===")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

acc_knn = accuracy_score(y_test, y_pred_knn)

print("Accuracy:", acc_knn)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))


# --- KNN Q2 ---

print("\n=== KNN Q2 ===")

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)

y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)

acc_knn_scaled = accuracy_score(
    y_test,
    y_pred_knn_scaled
)

print("Scaled Accuracy:", acc_knn_scaled)

# Unscaled KNN had an accuracy of 1.0000, while scaled KNN had an
# accuracy of 0.9333 on this test split.
#
# Therefore, scaling hurt KNN performance on this particular split.
# Even though KNN is distance-based and scaling is often useful,
# this result shows that scaling does not guarantee better accuracy
# on every dataset or every train/test split.


# --- KNN Q3 ---

print("\n=== KNN Q3 ===")

cv_scores = cross_val_score(
    KNeighborsClassifier(n_neighbors=5),
    X_train,
    y_train,
    cv=5
)

for fold, score in enumerate(cv_scores, start=1):
    print(f"Fold {fold} Accuracy: {score:.4f}")

print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")

# The single train/test split gave KNN an accuracy of 1.0000, while
# 5-fold cross-validation produced a mean accuracy of 0.9750 with a
# standard deviation of about 0.0333.
#
# I trust the cross-validation result more because it evaluates KNN
# across five different validation folds instead of depending on one
# especially favorable test split.
# --- KNN Q4 ---

print("\n=== KNN Q4 ===")

k_values = [
    1, 3, 5, 7, 9, 11, 13, 15
]

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

    print(
        f"k={k:2d}  "
        f"Mean CV Accuracy={mean_score:.4f}"
    )

    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print("\nSuggested k:", best_k)

# k=5 and k=7 both reached the highest mean cross-validation accuracy
# of 0.9750.
#
# The code selects k=5 because it is the first value that reaches the
# highest score. After k=7, the mean accuracy generally becomes slightly
# lower, so k=5 is a reasonable choice from the tested values.

# --- Classifier Evaluation Q1 ---

print("\n=== Classifier Evaluation Q1 ===")

cm = confusion_matrix(
    y_test,
    y_pred_knn
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris["target_names"])

disp.plot(cmap="Blues")

plt.title("KNN Classification of Iris Species")

plt.savefig(
    os.path.join(
        "outputs",
        "knn_confusion_matrix.png"
    )
)

plt.close()

# The confusion matrix shows that KNN classified most of the flowers
# correctly. Any values outside the main diagonal represent flowers
# that were confused with another Iris species.


# --- Decision Tree Q1 ---

print("\n=== Decision Tree Q1 ===")

tree = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

tree.fit(
    X_train,
    y_train
)

y_pred_tree = tree.predict(
    X_test
)

tree_acc = accuracy_score(
    y_test,
    y_pred_tree
)

print("Accuracy:", tree_acc)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_tree
    )
)

# The Decision Tree and KNN perform similarly on this dataset.
#
# Scaling would not normally change a Decision Tree because trees
# split features using thresholds rather than measuring distances.

# --- Logistic Regression Q1 ---

c_values = [
    0.01,
    1.0,
    100
]

for c in c_values:

    lr = LogisticRegression(
        C=c,
        max_iter=1000,
        solver="liblinear"
    )

    ovr_model = OneVsRestClassifier(lr)

    ovr_model.fit(
        X_train_scaled,
        y_train
    )

    # Each estimator is a fitted Logistic Regression model for
    # one class versus the remaining classes.
    #
    # For each fitted model, take the absolute value of every
    # coefficient and sum them. Then add those totals together
    # to get the total coefficient magnitude for this C value.
    total_coefficient_magnitude = sum(
        np.abs(model.coef_).sum()
        for model in ovr_model.estimators_
    )

    print(
        f"C = {c} | "
        f"Sum of absolute fitted coefficients = "
        f"{total_coefficient_magnitude:.4f}"
    )

# For C=0.01, the sum of the absolute fitted coefficients is about 1.9649.
# For C=1.0, it increases to about 12.4847.
# For C=100, it increases again to about 37.8903.
#
# This shows a clear trend: as C increases, regularization becomes weaker
# and the fitted Logistic Regression coefficients are allowed to become
# much larger.

# --- Digits Setup ---

digits = load_digits()

X_digits = digits["data"]
y_digits = digits["target"]
images = digits["images"]


# --- PCA Q1 ---

print("\n=== PCA Q1 ===")

print("X_digits shape:", X_digits.shape)
print("images shape :", images.shape)

fig, axes = plt.subplots(
    1,
    10,
    figsize=(15, 2)
)

for digit in range(10):

    idx = np.where(
        y_digits == digit
    )[0][0]

    axes[digit].imshow(
        images[idx],
        cmap="gray_r"
    )

    axes[digit].set_title(
        str(digit)
    )

    axes[digit].axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        "outputs",
        "sample_digits.png"
    )
)

plt.close()


# --- PCA Q2 ---

print("\n=== PCA Q2 ===")

pca = PCA()

pca.fit(X_digits)

scores = pca.transform(
    X_digits
)

plt.figure(
    figsize=(8, 6)
)

scatter = plt.scatter(
    scores[:, 0],
    scores[:, 1],
    c=y_digits,
    cmap="tab10",
    s=10
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("2D PCA Projection of Handwritten Digits")

plt.colorbar(
    scatter,
    label="Digit"
)

plt.savefig(
    os.path.join(
        "outputs",
        "pca_2d_projection.png"
    )
)

plt.close()

# Some digit groups are more separated than others.
# Digits such as 1, 2, 5, and 8 appear to have different positions
# in PCA space, although there is still overlap between classes.


# --- PCA Q3 ---

print("\n=== PCA Q3 ===")

cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

components_80 = (
    np.argmax(
        cumulative_variance >= 0.80
    ) + 1
)

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    range(
        1,
        len(cumulative_variance) + 1
    ),
    cumulative_variance
)

plt.axhline(
    0.80,
    linestyle="--"
)

plt.xlabel(
    "Number of Principal Components"
)

plt.ylabel(
    "Cumulative Explained Variance"
)

plt.title(
    "Cumulative Variance Explained by PCA"
)

plt.grid(True)

plt.savefig(
    os.path.join(
        "outputs",
        "pca_variance_explained.png"
    )
)

plt.close()

print(
    "Approximate number of components needed "
    "to explain at least 80% of the variance:",
    components_80
)


# The cumulative explained variance first reaches 80% at 13 principal
# components.
#
# The curve rises quickly through the first several components and then
# begins to level off around this region. After 13 components, each new
# component adds a smaller amount of explained variance than the earlier
# components.
#
# This means the first 13 components retain about 80% of the variation
# from the original 64 pixel features while using far fewer dimensions.

# --- PCA Q4 ---
print("\n=== PCA Q4 ===")


def reconstruct_digit(
    sample_idx,
    scores,
    pca,
    n_components
):
    reconstruction = pca.mean_.copy()

    for i in range(n_components):
        reconstruction += (
            scores[sample_idx, i]
            * pca.components_[i]
        )

    return reconstruction.reshape(
        8,
        8
    )


# Get one example each of digits 0-4.
sample_indices = [
    np.where(y_digits == digit)[0][0]
    for digit in range(5)
]

component_values = [
    2,
    5,
    15,
    40
]

# 5 rows:
# Original
# n = 2
# n = 5
# n = 15
# n = 40
#
# 5 columns:
# one example each of digits 0-4.

fig, axes = plt.subplots(
    5,
    5,
    figsize=(10, 10)
)


# Original images on the top row.
for col, sample_idx in enumerate(
    sample_indices
):

    axes[0, col].imshow(
        images[sample_idx],
        cmap="gray_r"
    )

    axes[0, col].set_title(
        f"Digit {y_digits[sample_idx]}"
    )

    axes[0, col].axis("off")


axes[0, 0].set_ylabel(
    "Original"
)


# Reconstruction rows.
for row, n_components in enumerate(
    component_values,
    start=1
):

    for col, sample_idx in enumerate(
        sample_indices
    ):

        reconstruction = reconstruct_digit(
            sample_idx,
            scores,
            pca,
            n_components
        )

        axes[row, col].imshow(
            reconstruction,
            cmap="gray_r"
        )

        axes[row, col].axis("off")

    # Clearly label each reconstruction row
    # with the number of PCA components used.
    axes[row, 0].set_ylabel(
        f"n = {n_components}"
    )


plt.suptitle(
    "Digit Reconstruction Using Different Numbers of PCA Components"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        "outputs",
        "pca_reconstructions.png"
    )
)

plt.close()

print(
    "Saved PCA reconstruction figure."
)


# With 2 components, the reconstructions are very rough because only
# about 28.5% of the original variance is being preserved.
#
# With 5 components, about 54.5% of the variance is preserved, so more
# of the digit shape becomes visible.
#
# With 15 components, about 83.5% of the variance is preserved and the
# digits are much clearer. This is also just beyond the 13-component
# point where the variance curve first reached 80%.
#
# With 40 components, about 98.8% of the variance is preserved, so the
# reconstructed digits look very close to the originals.
#
# This matches the explained-variance curve: most of the improvement
# happens earlier, and later components add progressively smaller details.
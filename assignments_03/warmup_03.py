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

# Fit only on the training data so information from the test set
# does not leak into the training process.
X_train_scaled = scaler.fit_transform(X_train)

# The test data uses the mean and standard deviation learned
# from the training data.
X_test_scaled = scaler.transform(X_test) # Scaling does importve performance and can more accurately descrive relationships between features on an even scale.

print("Column means of X_train_scaled:")
print(X_train_scaled.mean(axis=0))


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

# Scaling can improve KNN because KNN measures distance between points.
# If one feature has much larger values than another feature, it could
# have too much influence on the distance calculation.
# Scaling puts the features on comparable scales.


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

# Cross-validation tests the model on several different splits of the
# training data. This gives a better idea of whether the model performs
# consistently instead of depending on one split so it becomes more trust worthy the more runs you have that are consistent.


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
# Running through tests of each K value this result here would be the proven to be 5 because after mean accuracy decreases

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

    model = OneVsRestClassifier(
        LogisticRegression(
            C=c,
            max_iter=1000,
            solver="liblinear"
        )
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # OneVsRestClassifier fits one Logistic Regression model for
    # each class. Combine the coefficient arrays from those fitted
    # models so the total coefficient size can be calculated using
    # the exact expression requested in the assignment.
    model.coef_ = np.vstack([
        estimator.coef_
        for estimator in model.estimators_
    ])

    total_coefficient_size = np.abs(
        model.coef_
    ).sum()

    print(
        f"C = {c} | "
        f"Total coefficient size = "
        f"{total_coefficient_size:.4f}"
    )


# C controls the strength of regularization.
#
# A smaller C means stronger regularization, which restricts the
# coefficients more and usually produces a smaller total coefficient
# magnitude.
#
# A larger C means weaker regularization, so the model is allowed to
# use larger coefficients to fit the training data.
#
# Comparing the three printed totals shows how coefficient magnitude
# changes as regularization becomes weaker.
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


# The cumulative explained variance curve first reaches the 80%
# threshold at the number of components printed above.
#
# Therefore, that printed value is approximately how many principal
# components are needed to retain at least 80% of the total variation
# in the original 64-dimensional digit data.

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


# The top row shows the original digit images.
#
# The rows below show reconstructions using n = 2, 5, 15, and 40
# principal components.
#
# With only 2 components, the digits are recognizable only in a
# very rough way.
#
# At 5 components, more of the general digit shape appears.
#
# At 15 components, the digits become much clearer.
#
# At 40 components, the reconstruction is very close to the original.
#
# This matches the cumulative explained variance curve because using
# more principal components preserves more of the original variation.
#
# The large improvement by around 15 components also makes sense because
# the cumulative variance curve shows that much of the total variance
# has already been captured by that point.
#
# This shows that PCA can represent much of the original 64-dimensional
# pixel information using fewer dimensions.

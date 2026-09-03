import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


os.makedirs(
    "outputs",
    exist_ok=True
)


# --- Task 1: Load Dataset ---

spambase = fetch_ucirepo(
    id=94
)

X = spambase.data.features
y = spambase.data.targets

print(spambase.metadata)
print(spambase.variables)

print("\nDATASET OVERVIEW")

df = X.copy()

target_col = y.columns[0]

df[target_col] = y

print(
    f"Number of emails: {len(df)}"
)

print(
    f"Number of features including target: "
    f"{df.shape[1]}"
)

print("\nClass counts:")

print(
    df[target_col].value_counts()
)

print("\nClass percentages:")

print(
    df[target_col].value_counts(
        normalize=True
    ) * 100
)

spam_pct = (
    df[target_col].mean()
    * 100
)

ham_pct = (
    100 - spam_pct
)

print(
    f"\nSpam: {spam_pct:.2f}%"
)

print(
    f"Ham : {ham_pct:.2f}%"
)


# There are 4601 emails in the dataset.
#
# There are more ham emails than spam emails, so the classes are
# somewhat imbalanced.
#
# This matters because accuracy by itself could hide whether a model
# performs worse on the smaller spam class.


# --- Task 2: Explore Features ---

features_to_plot = [
    "word_freq_free",
    "char_freq_!",
    "capital_run_length_total"
]

for feature in features_to_plot:

    plt.figure(
        figsize=(6, 4)
    )

    df.boxplot(
        column=feature,
        by=target_col
    )

    plt.title(
        f"{feature} by Spam Label"
    )

    plt.suptitle("")

    plt.xlabel(
        "Spam Label (0=Ham, 1=Spam)"
    )

    plt.ylabel(feature)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            "outputs",
            f"{feature}_boxplot.png"
        )
    )

    plt.close()


print(
    "\nSaved boxplots to outputs/"
)

# The distributions contain many outliers, especially for features
# based on word frequency, character frequency, and capital letter runs.
#
# Some spam and ham emails overlap, but there are also differences
# in their distributions. These differences may help a classifier
# distinguish the two classes.


# --- Task 3: Train/Test Split ---

X = df.drop(
    columns=[target_col]
)

y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --- Scaling ---

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# Scaling matters for models such as KNN and Logistic Regression
# because the Spambase features use very different numerical scales.
#
# The scaler is fit only on training data to avoid data leakage.


# --- PCA ---

pca = PCA()

pca.fit(
    X_train_scaled
)

cumulative_variance = (
    pca.explained_variance_ratio_
    .cumsum()
)

n = (
    cumulative_variance >= 0.90
).argmax() + 1

print(
    "\nPCA Components Needed "
    "for 90% Variance:",
    n
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
    0.90,
    linestyle="--"
)

plt.xlabel(
    "Number of Components"
)

plt.ylabel(
    "Cumulative Explained Variance"
)

plt.title(
    "PCA Cumulative Explained Variance"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        "outputs",
        "pca_explained_variance.png"
    )
)

plt.close()


X_train_pca = (
    pca.transform(
        X_train_scaled
    )[:, :n]
)

X_test_pca = (
    pca.transform(
        X_test_scaled
    )[:, :n]
)


# --- KNN: Unscaled ---

print("\nKNN - UNSCALED")

knn_unscaled = KNeighborsClassifier(
    n_neighbors=5
)

knn_unscaled.fit(
    X_train,
    y_train
)

knn_unscaled_preds = (
    knn_unscaled.predict(
        X_test
    )
)

knn_unscaled_acc = accuracy_score(
    y_test,
    knn_unscaled_preds
)

print(
    "Accuracy:",
    knn_unscaled_acc
)

print(
    classification_report(
        y_test,
        knn_unscaled_preds
    )
)


# --- KNN: Scaled ---

print("\nKNN - SCALED")

knn_scaled = KNeighborsClassifier(
    n_neighbors=5
)

knn_scaled.fit(
    X_train_scaled,
    y_train
)

knn_scaled_preds = (
    knn_scaled.predict(
        X_test_scaled
    )
)

knn_scaled_acc = accuracy_score(
    y_test,
    knn_scaled_preds
)

print(
    "Accuracy:",
    knn_scaled_acc
)

print(
    classification_report(
        y_test,
        knn_scaled_preds
    )
)


# --- KNN: PCA ---

print("\nKNN - PCA")

knn_pca = KNeighborsClassifier(
    n_neighbors=5
)

knn_pca.fit(
    X_train_pca,
    y_train
)

knn_pca_preds = (
    knn_pca.predict(
        X_test_pca
    )
)

knn_pca_acc = accuracy_score(
    y_test,
    knn_pca_preds
)

print(
    "Accuracy:",
    knn_pca_acc
)

print(
    classification_report(
        y_test,
        knn_pca_preds
    )
)

# Comparing the three KNN models shows whether scaling and PCA
# improve a distance-based model.
#
# Scaling can help because KNN calculates distances between samples.
# PCA may help if reducing redundant dimensions makes those distances
# more meaningful, but PCA can also remove information that is useful
# for classification.


# --- Decision Tree Depth Comparison ---

print(
    "\nDECISION TREE DEPTH COMPARISON"
)

depths = [
    3,
    5,
    10,
    None
]

tree_depth_results = {}

for depth in depths:

    temp_tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    temp_tree.fit(
        X_train,
        y_train
    )

    train_acc = accuracy_score(
        y_train,
        temp_tree.predict(
            X_train
        )
    )

    test_acc = accuracy_score(
        y_test,
        temp_tree.predict(
            X_test
        )
    )

    tree_depth_results[depth] = (
        train_acc,
        test_acc
    )

    print(
        f"Depth={depth} | "
        f"Train={train_acc:.4f} | "
        f"Test={test_acc:.4f}"
    )


best_tree_depth = 5

# I chose max_depth=5 because it provides a balance between training
# and test performance. Deeper trees can continue improving training
# accuracy while giving little or no improvement on the test data,
# which can be a sign of overfitting.
#
# A depth of 5 keeps the model simpler while still giving strong
# test performance.


tree = DecisionTreeClassifier(
    max_depth=best_tree_depth,
    random_state=42
)

tree.fit(
    X_train,
    y_train
)

tree_preds = tree.predict(
    X_test
)

tree_acc = accuracy_score(
    y_test,
    tree_preds
)

print(
    "\nFINAL DECISION TREE"
)

print(
    "Accuracy:",
    tree_acc
)

print(
    classification_report(
        y_test,
        tree_preds
    )
)


# --- Decision Tree Feature Importances ---

tree_importances = pd.Series(
    tree.feature_importances_,
    index=X.columns
).sort_values(
    ascending=False
)

print(
    "\nTOP 10 DECISION TREE FEATURE IMPORTANCES"
)

print(
    tree_importances.head(10)
)


# --- Random Forest ---

rf = RandomForestClassifier(
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

rf_preds = rf.predict(
    X_test
)

rf_acc = accuracy_score(
    y_test,
    rf_preds
)

print(
    "\nRANDOM FOREST"
)

print(
    "Accuracy:",
    rf_acc
)

print(
    classification_report(
        y_test,
        rf_preds
    )
)


# --- Random Forest Feature Importances ---

rf_importances = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(
    ascending=False
)

print(
    "\nTOP 10 RANDOM FOREST FEATURE IMPORTANCES"
)

print(
    rf_importances.head(10)
)


top_10_rf = rf_importances.head(
    10
).sort_values()


plt.figure(
    figsize=(8, 6)
)

plt.barh(
    top_10_rf.index,
    top_10_rf.values
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 10 Random Forest Feature Importances"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        "outputs",
        "random_forest_feature_importances.png"
    )
)

plt.close()

# Feature importance tells us which variables the tree-based models
# relied on most when making splits.
#
# Random Forest importance is useful because it combines information
# across many trees instead of depending on one individual tree.


# --- Logistic Regression: Scaled ---

lr_scaled = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="liblinear"
)

lr_scaled.fit(
    X_train_scaled,
    y_train
)

lr_scaled_preds = (
    lr_scaled.predict(
        X_test_scaled
    )
)

lr_scaled_acc = accuracy_score(
    y_test,
    lr_scaled_preds
)

print(
    "\nLOGISTIC REGRESSION - SCALED"
)

print(
    "Accuracy:",
    lr_scaled_acc
)

print(
    classification_report(
        y_test,
        lr_scaled_preds
    )
)


# --- Logistic Regression: PCA ---

lr_pca = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="liblinear"
)

lr_pca.fit(
    X_train_pca,
    y_train
)

lr_pca_preds = (
    lr_pca.predict(
        X_test_pca
    )
)

lr_pca_acc = accuracy_score(
    y_test,
    lr_pca_preds
)

print(
    "\nLOGISTIC REGRESSION - PCA"
)

print(
    "Accuracy:",
    lr_pca_acc
)

print(
    classification_report(
        y_test,
        lr_pca_preds
    )
)


# PCA comparison:
#
# Comparing scaled Logistic Regression with PCA Logistic Regression
# tells us whether reducing the dimensionality improves prediction.
#
# If PCA accuracy is higher, the lower-dimensional representation
# may be removing noise or redundant information.
#
# If scaled Logistic Regression is higher, some predictive information
# may have been lost when PCA reduced the features.


# --- Test Set Model Comparison ---

results = {
    "KNN Unscaled":
        knn_unscaled_acc,

    "KNN Scaled":
        knn_scaled_acc,

    "KNN PCA":
        knn_pca_acc,

    "Decision Tree":
        tree_acc,

    "Random Forest":
        rf_acc,

    "Logistic Regression Scaled":
        lr_scaled_acc,

    "Logistic Regression PCA":
        lr_pca_acc
}


print(
    "\nTEST SET MODEL ACCURACIES"
)

for name, score in results.items():

    print(
        f"{name:30s}: "
        f"{score:.4f}"
    )


best_model_name = max(
    results,
    key=results.get
)

print(
    "\nBest Test Model:",
    best_model_name
)


# --- Best Model Confusion Matrix ---

prediction_lookup = {
    "KNN Unscaled":
        knn_unscaled_preds,

    "KNN Scaled":
        knn_scaled_preds,

    "KNN PCA":
        knn_pca_preds,

    "Decision Tree":
        tree_preds,

    "Random Forest":
        rf_preds,

    "Logistic Regression Scaled":
        lr_scaled_preds,

    "Logistic Regression PCA":
        lr_pca_preds
}


best_preds = prediction_lookup[
    best_model_name
]


cm = confusion_matrix(
    y_test,
    best_preds
)

tn, fp, fn, tp = cm.ravel()


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Ham",
        "Spam"
    ]
)

disp.plot()

plt.title(
    f"{best_model_name} Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        "outputs",
        "best_model_confusion_matrix.png"
    )
)

plt.close()


print(
    "\nBest Model Error Counts"
)

print(
    "False Positives "
    "(ham marked spam):",
    fp
)

print(
    "False Negatives "
    "(spam marked ham):",
    fn
)


if fp > fn:

    print(
        "The model makes more "
        "false positives."
    )

elif fn > fp:

    print(
        "The model makes more "
        "false negatives."
    )

else:

    print(
        "The model makes the same "
        "number of each error."
    )


# A false positive means a real ham email is incorrectly sent to spam.
# A false negative means a spam email gets through as ham.
#
# I would be especially concerned about false positives because an
# important real email could be hidden in the spam folder.
#
# However, false negatives also matter because allowing too much spam
# through defeats the purpose of the classifier.


# --- Task 4: Cross Validation ---

print("\nTASK 4 - CROSS VALIDATION")

models = {
    "KNN Unscaled": KNeighborsClassifier(
        n_neighbors=5
    ),

    "KNN Scaled": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),

    "KNN PCA": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n)),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=best_tree_depth,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        random_state=42
    ),

    "Logistic Regression Scaled": Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="liblinear"
            )
        )
    ]),

    "Logistic Regression PCA": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n)),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="liblinear"
            )
        )
    ])
}


cv_results = {}

for name, classifier in models.items():

    scores = cross_val_score(
        classifier,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    cv_results[name] = scores.mean()

    print(f"\n{name}")

    for fold, score in enumerate(scores, start=1):
        print(f"Fold {fold}: {score:.4f}")

    print(f"Mean Accuracy: {scores.mean():.4f}")
    print(f"Standard Deviation: {scores.std():.4f}")


best_cv_model = max(
    cv_results,
    key=cv_results.get
)

print("\nBest Cross-Validated Model:", best_cv_model)

# All classifiers from Task 3 are tested again using 5-fold
# cross-validation. Comparing the mean accuracies gives a more reliable
# comparison than using only one train/test split. The standard deviation
# also shows how consistent each model is across the folds.


# --- Task 5: Tree Pipeline ---

tree_pipeline = Pipeline([
    (
        "classifier",
        RandomForestClassifier(
            random_state=42
        )
    )
])

tree_pipeline.fit(
    X_train,
    y_train
)

tree_pipeline_preds = tree_pipeline.predict(
    X_test
)

tree_pipeline_acc = accuracy_score(
    y_test,
    tree_pipeline_preds
)

print("\nRandom Forest Pipeline")

print("Manual Random Forest Accuracy:", rf_acc)
print("Pipeline Random Forest Accuracy:", tree_pipeline_acc)

print(
    "Results Match:",
    np.isclose(
        rf_acc,
        tree_pipeline_acc
    )
)

print(
    classification_report(
        y_test,
        tree_pipeline_preds
    )
)

# Same model and data as the
# manual Random Forest, so its accuracy should match the earlier result.
# Random Forest does not require scaling or PCA, so the pipeline only
# needs the classifier step.


# --- Task 5: Non-Tree Pipeline ---

non_tree_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=n)),
    (
        "classifier",
        LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="liblinear"
        )
    )
])

non_tree_pipeline.fit(
    X_train,
    y_train
)

non_tree_pipeline_preds = non_tree_pipeline.predict(
    X_test
)

non_tree_pipeline_acc = accuracy_score(
    y_test,
    non_tree_pipeline_preds
)

print("\nLogistic Regression PCA Pipeline")

print(
    "Manual Logistic Regression PCA Accuracy:",
    lr_pca_acc
)

print(
    "Pipeline Logistic Regression PCA Accuracy:",
    non_tree_pipeline_acc
)

print(
    "Results Match:",
    np.isclose(
        lr_pca_acc,
        non_tree_pipeline_acc
    )
)

print(
    classification_report(
        y_test,
        non_tree_pipeline_preds
    )
)

# Logistic Regression approach: scaling, PCA, and then classification.
# Because the same preprocessing and classifier are being used, the
# pipeline result should match or be close to the manual result.
#
# Even if PCA does not improve Logistic Regression accuracy, this pipeline
# still demonstrates how PCA preprocessing can be included correctly
# before the classifier.


# --- Final Summary ---

print(
    "\nFINAL SUMMARY"
)

print(
    "Best Cross-Validated Model:",
    best_cv_model
)

print(
    "Best Tree-Based Model:",
    best_tree_name
)

print(
    "Best Non-Tree Model:",
    best_non_tree_name
)

print(
    "Best Test Model:",
    best_model_name
)


# Overall, the models show that preprocessing affects different
# classifiers differently.
#
# KNN is sensitive to feature scale because it uses distances, so
# comparing unscaled, scaled, and PCA versions shows how preprocessing
# changes its performance.
#
# Logistic Regression also benefits from scaling. PCA may improve or
# reduce its accuracy depending on whether the reduced components keep
# the information most useful for separating spam from ham.
#
# Decision Trees and Random Forests do not require scaling because they
# make threshold-based splits rather than distance calculations.
#
# Random Forest tends to perform strongly because many different trees
# are combined, which usually gives a more stable classifier than one
# Decision Tree.
#
# The confusion matrix tells us which kind of mistake the best model
# makes more often. False positives are ham emails incorrectly marked
# as spam, while false negatives are spam emails incorrectly allowed
# through as ham.

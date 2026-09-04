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
    "\nNumber of PCA components needed "
    "to reach at least 90% cumulative variance:",
    n
)


# n is the first number of principal components where the cumulative
# explained variance reaches or exceeds 90%.
#
# I keep X_train_scaled and X_test_scaled because some classifiers
# are evaluated using all of the scaled features.
#
# I also create X_train_pca and X_test_pca using only the first n
# principal components so I can compare the full scaled feature set
# with the reduced PCA representation.


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
        "pca_variance_explained.png"
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


# Choose the depth from the results above instead of hard-coding it.
# The highest test accuracy is preferred. If two depths tie, the
# shallower tree is chosen because it is less complex and less likely
# to overfit.
depth_complexity = {
    3: 3,
    5: 5,
    10: 10,
    None: float("inf")
}

best_tree_depth = max(
    depths,
    key=lambda depth: (
        tree_depth_results[depth][1],
        -depth_complexity[depth]
    )
)

selected_train_acc = tree_depth_results[
    best_tree_depth
][0]

selected_test_acc = tree_depth_results[
    best_tree_depth
][1]

print(
    "\nSelected Decision Tree Depth:",
    best_tree_depth
)

print(
    "Selected Depth Training Accuracy:",
    f"{selected_train_acc:.4f}"
)

print(
    "Selected Depth Test Accuracy:",
    f"{selected_test_acc:.4f}"
)

# The production depth is chosen directly from the depth comparison
# printed above. I first use test accuracy because it shows how well
# each depth generalizes to unseen data.
#
# If two depths have the same test accuracy, I choose the shallower
# one because a simpler tree is less likely to overfit. Deeper trees
# can keep increasing training accuracy without improving test accuracy.


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


# Directly compare the top 10 features from both tree models.
tree_top_features = set(
    tree_importances.head(10).index
)

rf_top_features = set(
    rf_importances.head(10).index
)

common_features = (
    tree_top_features
    .intersection(rf_top_features)
)

tree_only_features = (
    tree_top_features
    - rf_top_features
)

rf_only_features = (
    rf_top_features
    - tree_top_features
)

print(
    "\nFeatures in both top 10 lists:",
    common_features
)

print(
    "Decision Tree-only top features:",
    tree_only_features
)

print(
    "Random Forest-only top features:",
    rf_only_features
)

# The common_features output shows exactly where the Decision Tree and
# Random Forest agree about which features matter most.
#
# The tree_only_features and rf_only_features outputs show where they
# disagree. This is expected because the Decision Tree gets importance
# from one tree, while the Random Forest averages importance across
# many different trees.


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
        "feature_importances.png"
    )
)

plt.close()

# This bar chart shows the 10 features with the highest Random Forest
# feature-importance values.
#
# Random Forest calculates importance across many trees, so these
# values represent which features were most useful for making splits
# across the entire ensemble.
#
# The figure is saved as outputs/feature_importances.png exactly as
# required by the assignment.

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


# Directly compare Logistic Regression with and without PCA.
print(
    "\nLOGISTIC REGRESSION COMPARISON"
)

print(
    "Scaled Logistic Regression:",
    f"{lr_scaled_acc:.4f}"
)

print(
    "PCA Logistic Regression:",
    f"{lr_pca_acc:.4f}"
)

if lr_pca_acc > lr_scaled_acc:

    print(
        "PCA improved Logistic Regression "
        "on this test split."
    )

elif lr_pca_acc < lr_scaled_acc:

    print(
        "PCA reduced Logistic Regression "
        "on this test split."
    )

else:

    print(
        "PCA made no difference to Logistic Regression "
        "on this test split."
    )

# This comparison uses the two Logistic Regression results printed
# above, so the PCA and non-PCA versions are directly compared in the
# same final Task 3 classifier summary.


best_model_name = max(
    results,
    key=results.get
)

print(
    "\nBest Test Model:",
    best_model_name
)


# --- Best Model Confusion Matrix ---

print(
    "\nBEST-PERFORMING CLASSIFIER:",
    best_model_name
)

print(
    "Test Accuracy:",
    f"{results[best_model_name]:.4f}"
)


# best_model_name was selected from the Task 3 comparison using the
# highest test-set accuracy.
#
# The predictions below are therefore specifically from that
# best-performing classifier.

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
    f"\nERROR ANALYSIS FOR BEST MODEL: "
    f"{best_model_name}"
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
        f"{best_model_name} makes more false positives "
        f"({fp}) than false negatives ({fn})."
    )

elif fn > fp:

    print(
        f"{best_model_name} makes more false negatives "
        f"({fn}) than false positives ({fp})."
    )

else:

    print(
        f"{best_model_name} makes the same number of "
        f"false positives and false negatives ({fp})."
    )


# This error analysis belongs specifically to the best-performing
# classifier selected from the Task 3 test-set comparison.
#
# A false positive means a legitimate ham email was incorrectly
# classified as spam. A false negative means a spam email was
# incorrectly classified as legitimate ham.
#
# The printed counts above directly show which type of error occurs
# more often for the selected best model.
#
# For a spam filter, I would be more concerned about false positives
# because a legitimate and potentially important email could be moved
# into the spam folder and missed by the user.

# --- Task 4: Cross Validation ---

print("\nTASK 4 - CROSS VALIDATION")

cv_results = {}

# Use the same seven classifier/preprocessing combinations from Task 3.
# The scaled models include StandardScaler, and the PCA models include
# both StandardScaler and the same number of PCA components, n, that
# was selected in Task 3. The Decision Tree also uses the depth selected
# from the Task 3 depth comparison.
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

for name, classifier in models.items():

    scores = cross_val_score(
        classifier,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    cv_results[name] = {
        "mean": scores.mean(),
        "std": scores.std()
    }

    print(f"\n{name}")

    for fold, score in enumerate(scores, start=1):
        print(f"Fold {fold}: {score:.4f}")

    print(f"Mean Accuracy: {scores.mean():.4f}")
    print(f"Standard Deviation: {scores.std():.4f}")


best_cv_model = max(
    cv_results,
    key=lambda name: cv_results[name]["mean"]
)

most_stable_model = min(
    cv_results,
    key=lambda name: cv_results[name]["std"]
)

print(
    "\nMost Accurate Model:",
    best_cv_model,
    f"({cv_results[best_cv_model]['mean']:.4f})"
)

print(
    "Most Stable Model:",
    most_stable_model,
    f"(std={cv_results[most_stable_model]['std']:.4f})"
)

# The model with the highest mean CV accuracy is the most accurate overall.
# The model with the lowest standard deviation is the most stable because
# its accuracy changes the least between folds.

# These are the same seven classifier/preprocessing combinations tested
# in Task 3. Each one is now evaluated using 5-fold cross-validation.
#
# I compare their mean CV accuracy to see which performs best across
# several different validation splits instead of relying on one test split.
#
# I also compare their standard deviations. A smaller standard deviation
# means the model's performance was more consistent between folds.
#
# The best cross-validated model may not be the same as the best model
# from the single test split because cross-validation evaluates the model
# across several different partitions of the training data.


# --- Task 5: Pipelines ---

# Find the best tree-based model from cross-validation.

tree_model_names = [
    "Decision Tree",
    "Random Forest"
]

best_tree_name = max(
    tree_model_names,
    key=lambda name: cv_results[name]["mean"]
)



# Find the best non-tree-based model from cross-validation.

non_tree_model_names = [
    "KNN Unscaled",
    "KNN Scaled",
    "KNN PCA",
    "Logistic Regression Scaled",
    "Logistic Regression PCA"
]

best_non_tree_name = max(
    non_tree_model_names,
    key=lambda name: cv_results[name]["mean"]
)


print("\nBest Tree-Based CV Model:", best_tree_name)
print("Best Non-Tree CV Model:", best_non_tree_name)


# --- Best Tree Pipeline ---

if best_tree_name == "Random Forest":

    tree_pipeline = Pipeline([
        (
            "classifier",
            RandomForestClassifier(
                random_state=42
            )
        )
    ])

elif best_tree_name == "Decision Tree":

    tree_pipeline = Pipeline([
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=best_tree_depth,
                random_state=42
            )
        )
    ])

else:

    raise ValueError(
        "Unknown tree model: "
        f"{best_tree_name}"
    )


# This pipeline exactly mirrors the Task 3 tree configuration.
# Random Forest used the original unscaled features, and the final
# Decision Tree used the selected best_tree_depth on the original
# unscaled features. No scaling or PCA is added because Task 3 did
# not use those steps for either tree model.

print(
    "Tree Pipeline Steps:",
    list(tree_pipeline.named_steps.keys())
)


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


print(
    "\nBEST TREE PIPELINE:",
    best_tree_name
)

print(
    "Accuracy:",
    tree_pipeline_acc
)

print(
    classification_report(
        y_test,
        tree_pipeline_preds
    )
)


# The best tree-based classifier is selected using mean cross-validation
# accuracy. Tree models do not require scaling because they make splits
# based on feature thresholds instead of distances.


# --- Best Non-Tree Pipeline ---

if best_non_tree_name == "KNN Unscaled":

    non_tree_pipeline = Pipeline([
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ])

elif best_non_tree_name == "KNN Scaled":

    non_tree_pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ])

elif best_non_tree_name == "KNN PCA":

    non_tree_pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "pca",
            PCA(
                n_components=n
            )
        ),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ])

elif best_non_tree_name == "Logistic Regression Scaled":

    non_tree_pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="liblinear"
            )
        )
    ])

elif best_non_tree_name == "Logistic Regression PCA":

    non_tree_pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "pca",
            PCA(
                n_components=n
            )
        ),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="liblinear"
            )
        )
    ])

else:

    raise ValueError(
        "Unknown non-tree model: "
        f"{best_non_tree_name}"
    )


# Each possible pipeline above exactly matches the preprocessing/model
# combination used for that same named model in Task 3:
#
# KNN Unscaled = KNN only
# KNN Scaled = StandardScaler + KNN
# KNN PCA = StandardScaler + PCA(n) + KNN
# Logistic Regression Scaled = StandardScaler + Logistic Regression
# Logistic Regression PCA = StandardScaler + PCA(n) + Logistic Regression

print(
    "Non-Tree Pipeline Steps:",
    list(non_tree_pipeline.named_steps.keys())
)


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


print(
    "\nBEST NON-TREE PIPELINE:",
    best_non_tree_name
)

print(
    "Accuracy:",
    non_tree_pipeline_acc
)

print(
    classification_report(
        y_test,
        non_tree_pipeline_preds
    )
)


# The best non-tree classifier is selected using mean cross-validation
# accuracy. After the model name is selected, the matching branch above
# rebuilds the exact preprocessing/model combination used in Task 3.
#
# This removes ambiguity because each possible model name has its own
# explicit Pipeline definition rather than sharing a generic pipeline.


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

# Logistic Regression PCA vs. non-PCA was compared directly in the
# Task 3 model comparison section.
#
# Decision Tree vs. Random Forest feature overlap was also compared
# directly in the Task 3 feature-importance section.


# Scaling changed KNN because KNN depends on distances between features.
# PCA changed the result again by replacing the original features with
# lower-dimensional principal components.

# For spam filtering, I would prefer more false negatives over false positives.
# A false negative allows some spam into the inbox, which is annoying, but a
# false positive sends a real email to spam and could cause the user to miss
# something important.

# The two pipelines are different because the classifiers have different
# preprocessing needs.
#
# Tree-based models do not depend on feature distances, so scaling is not
# needed.
#
# KNN and Logistic Regression can be affected by feature scale, so their
# pipelines include StandardScaler when that preprocessing was used.
#
# If the selected non-tree model used PCA earlier, PCA is included here too
# so the pipeline matches the same preprocessing/model combination.


manual_tree_accuracy = results[best_tree_name]
manual_non_tree_accuracy = results[best_non_tree_name]

print(
    "Tree Pipeline Matches Earlier Result:",
    np.isclose(
        tree_pipeline_acc,
        manual_tree_accuracy
    )
)

print(
    "Non-Tree Pipeline Matches Earlier Result:",
    np.isclose(
        non_tree_pipeline_acc,
        manual_non_tree_accuracy
    )
)

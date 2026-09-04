import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests

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


# --- Task 1.1: Load Dataset ---

COLUMN_NAMES = [
    "word_freq_make",
    "word_freq_address",
    "word_freq_all",
    "word_freq_3d",
    "word_freq_our",
    "word_freq_over",
    "word_freq_remove",
    "word_freq_internet",
    "word_freq_order",
    "word_freq_mail",
    "word_freq_receive",
    "word_freq_will",
    "word_freq_people",
    "word_freq_report",
    "word_freq_addresses",
    "word_freq_free",
    "word_freq_business",
    "word_freq_email",
    "word_freq_you",
    "word_freq_credit",
    "word_freq_your",
    "word_freq_font",
    "word_freq_000",
    "word_freq_money",
    "word_freq_hp",
    "word_freq_hpl",
    "word_freq_george",
    "word_freq_650",
    "word_freq_lab",
    "word_freq_labs",
    "word_freq_telnet",
    "word_freq_857",
    "word_freq_data",
    "word_freq_415",
    "word_freq_85",
    "word_freq_technology",
    "word_freq_1999",
    "word_freq_parts",
    "word_freq_pm",
    "word_freq_direct",
    "word_freq_cs",
    "word_freq_meeting",
    "word_freq_original",
    "word_freq_project",
    "word_freq_re",
    "word_freq_edu",
    "word_freq_table",
    "word_freq_conference",
    "char_freq_;",
    "char_freq_(",
    "char_freq_[",
    "char_freq_!",
    "char_freq_$",
    "char_freq_#",
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total",
    "spam_label",
]


url = (
    "https://archive.ics.uci.edu/"
    "ml/machine-learning-databases/"
    "spambase/spambase.data"
)

response = requests.get(url)

response.raise_for_status()

df = pd.read_csv(
    BytesIO(response.content),
    header=None
)

df.columns = COLUMN_NAMES


# --- Task 1.2: Class Balance ---

print(
    "Number of emails:",
    len(df)
)

print(
    "\nClass counts:"
)

print(
    df["spam_label"].value_counts()
)

print(
    "\nClass percentages:"
)

print(
    df["spam_label"].value_counts(
        normalize=True
    ) * 100
)


# There are 4601 emails in the dataset, and there are more ham
# emails than spam emails. The classes are therefore somewhat
# imbalanced rather than being split evenly.
#
# Because ham is the larger class, raw accuracy by itself can be
# misleading. A classifier could predict ham often and still get
# a fairly high accuracy while performing worse at identifying spam.
#
# Because of this class imbalance, I will also look at the
# classification reports and confusion matrix rather than using
# accuracy alone to judge the classifiers.


# --- Task 1.3: Feature Distributions ---

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
        by="spam_label"
    )

    plt.title(
        f"{feature} by Spam Label"
    )

    plt.suptitle("")

    plt.xlabel(
        "Spam Label (0=Ham, 1=Spam)"
    )

    plt.ylabel(
        feature
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            "outputs",
            f"{feature}_boxplot.png"
        )
    )

    plt.close()


print(
    "\nSaved three feature boxplots to outputs/"
)


# The boxplots show overlap between spam and ham, so none of these
# individual features completely separates the two classes.
#
# However, the distributions are different between the classes.
# Features such as the frequency of "free", exclamation marks,
# and capital-letter runs can therefore still help distinguish
# spam from ham when they are combined with the other features.
#
# The differences are useful but not enough for one feature alone
# to identify whether an email is spam.


# --- Task 1.4: Raw Feature Scales ---

# Many word-frequency features contain a large number of zeros
# because most emails do not contain every word tracked by the
# dataset. For example, many emails never contain the word "free".
#
# The numeric scales vary because the features measure different
# things. Word and character frequencies are small proportions,
# while capital_run_length_total is a count that can reach much
# larger values.
#
# This matters for KNN because KNN calculates distances between
# samples. A feature with values in the hundreds or thousands
# could have much more influence than a feature measured as a
# small fraction.
#
# Because of this, Task 3 will compare KNN using the raw features
# and the scaled features. PCA will also be performed only after
# the data has been scaled.


# --- Task 2.1: Train/Test Split ---

X = df.drop(
    columns=["spam_label"]
)

y = df["spam_label"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# I use stratify=y so the spam and ham proportions remain similar
# in the training and test sets.


# --- Task 2.1: Scaling ---

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# The scaler is fit only on X_train so information from X_test
# does not leak into the training process.
#
# I keep X_train and X_test because Task 3 requires KNN on the
# unscaled data.
#
# I also keep X_train_scaled and X_test_scaled because Task 3
# requires scaled KNN and scaled Logistic Regression.


# --- Task 2.2: PCA ---

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
    "to first reach at least 90% explained variance:",
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
    "Number of Principal Components"
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


# n is the first number of principal components where the
# cumulative explained variance reaches or exceeds 90%.


# --- Task 2.3: PCA Transform ---

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


# I keep both versions of the data.
#
# X_train_scaled and X_test_scaled contain all scaled features.
# X_train_pca and X_test_pca contain only the first n principal
# components needed to preserve at least 90% of the variance.
#
# Task 3 will compare classifiers using these two representations.


# --- Task 3.1: KNN Unscaled ---

print(
    "\nKNN - UNSCALED"
)

knn_unscaled = KNeighborsClassifier(
    n_neighbors=5
)

knn_unscaled.fit(
    X_train,
    y_train
)

knn_unscaled_preds = knn_unscaled.predict(
    X_test
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


# --- Task 3.1: KNN Scaled ---

print(
    "\nKNN - SCALED"
)

knn_scaled = KNeighborsClassifier(
    n_neighbors=5
)

knn_scaled.fit(
    X_train_scaled,
    y_train
)

knn_scaled_preds = knn_scaled.predict(
    X_test_scaled
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


# --- Task 3.1: KNN PCA ---

print(
    "\nKNN - PCA"
)

knn_pca = KNeighborsClassifier(
    n_neighbors=5
)

knn_pca.fit(
    X_train_pca,
    y_train
)

knn_pca_preds = knn_pca.predict(
    X_test_pca
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


print(
    "\nKNN SCALED VS PCA"
)

print(
    "Scaled Accuracy:",
    knn_scaled_acc
)

print(
    "PCA Accuracy:",
    knn_pca_acc
)


if knn_scaled_acc > knn_pca_acc:

    print(
        "Scaled KNN performed better than PCA KNN."
    )

elif knn_pca_acc > knn_scaled_acc:

    print(
        "PCA KNN performed better than scaled KNN."
    )

else:

    print(
        "Scaled KNN and PCA KNN had the same accuracy."
    )


# This directly compares KNN using all scaled features with KNN
# using the PCA-reduced features.
#
# If PCA performs worse, reducing the dimensions removed some
# useful classification information.
#
# If PCA performs better, the reduced representation may have
# removed redundant information that was hurting distance calculations.


# --- Task 3.1: Decision Tree Depth Comparison ---

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
        f"Train Accuracy={train_acc:.4f} | "
        f"Test Accuracy={test_acc:.4f}"
    )


# Choose the depth with the highest test accuracy.
# Because depths is ordered from shallower to deeper, if two
# depths have the same test accuracy, the shallower one is kept.

best_tree_depth = max(
    depths,
    key=lambda depth:
        tree_depth_results[depth][1]
)


print(
    "\nSelected Decision Tree Depth:",
    best_tree_depth
)


# As depth increases, training accuracy generally increases because
# the tree can create more detailed rules for the training examples.
#
# If training accuracy continues increasing while test accuracy stops
# improving or decreases, the deeper tree is overfitting.
#
# I use the depth printed above because it produced the highest test
# accuracy among the required depths. This gives stronger performance
# on unseen data instead of selecting the tree only because it fits
# the training data more closely.


# --- Final Decision Tree ---

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


# --- Task 3.1: Random Forest ---

rf = RandomForestClassifier(
    n_estimators=100,
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


# --- Task 3.1: Logistic Regression Scaled ---

lr_scaled = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="liblinear"
)

lr_scaled.fit(
    X_train_scaled,
    y_train
)

lr_scaled_preds = lr_scaled.predict(
    X_test_scaled
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


# --- Task 3.1: Logistic Regression PCA ---

lr_pca = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="liblinear"
)

lr_pca.fit(
    X_train_pca,
    y_train
)

lr_pca_preds = lr_pca.predict(
    X_test_pca
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


print(
    "\nLOGISTIC REGRESSION SCALED VS PCA"
)

print(
    "Scaled Accuracy:",
    lr_scaled_acc
)

print(
    "PCA Accuracy:",
    lr_pca_acc
)


if lr_scaled_acc > lr_pca_acc:

    print(
        "Scaled Logistic Regression performed "
        "better than PCA Logistic Regression."
    )

elif lr_pca_acc > lr_scaled_acc:

    print(
        "PCA Logistic Regression performed "
        "better than scaled Logistic Regression."
    )

else:

    print(
        "Scaled and PCA Logistic Regression "
        "had the same accuracy."
    )


# This directly compares Logistic Regression using all scaled
# features with Logistic Regression using the PCA-reduced features.
#
# If PCA performs worse, dimensionality reduction removed some
# information that was useful for predicting spam.
#
# If PCA performs better, the reduced components may have removed
# redundant information while preserving the most useful variation.


# --- Task 3.2: Classifier Comparison ---

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
        lr_pca_acc,
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
    "\nBest-performing classifier:",
    best_model_name
)

print(
    "Best test accuracy:",
    results[best_model_name]
)


# The classifier printed above as the best-performing classifier
# has the highest test accuracy in this comparison.
#
# For KNN and Logistic Regression, the scaled and PCA accuracies
# printed above directly show whether PCA helped or hurt each model.
# This lets me compare the PCA result with the Task 2 idea that PCA
# may remove redundant dimensions but can also remove useful information.
#
# For a spam filter, I would not use accuracy as the only metric.
# I would rather minimize false positives than false negatives.
#
# A false positive sends a legitimate email to the spam folder,
# which could cause the user to miss something important.
# A false negative allows spam into the inbox, which is annoying,
# but it is usually less costly than hiding a legitimate email.


# --- Task 3.3: Best Model Confusion Matrix ---

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
        lr_pca_preds,
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
    "\nERROR ANALYSIS FOR BEST MODEL:",
    best_model_name
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
        "The best-performing classifier makes "
        "more false positives than false negatives."
    )

elif fn > fp:

    print(
        "The best-performing classifier makes "
        "more false negatives than false positives."
    )

else:

    print(
        "The best-performing classifier makes "
        "the same number of false positives "
        "and false negatives."
    )


# This confusion matrix uses predictions from the classifier that
# had the highest test accuracy in Task 3.2.
#
# The printed false-positive and false-negative counts directly
# identify which error the best-performing classifier makes more often.
#
# For this spam-filtering problem, false positives are the error I
# would be more concerned about because they hide legitimate email.


# --- Task 3.4: Decision Tree Feature Importances ---

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


# --- Task 3.4: Random Forest Feature Importances ---

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


tree_top_features = set(
    tree_importances.head(10).index
)

rf_top_features = set(
    rf_importances.head(10).index
)

common_features = (
    tree_top_features
    .intersection(
        rf_top_features
    )
)


print(
    "\nFeatures appearing in both top 10 lists:"
)

print(
    common_features
)


top_10_rf = rf_importances.head(
    10
).sort_values()


plt.figure(
    figsize=(9, 6)
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


# The common_features output shows which features both the
# Decision Tree and Random Forest place in their top 10.
# Those shared features are where the two models most clearly
# agree about what is important for identifying spam.
#
# The lists do not have to be identical because the Decision Tree
# calculates importance from one tree, while the Random Forest
# averages importance across 100 trees.
#
# Features involving unusual punctuation, spam-related words,
# and capitalization make sense as important features because
# those are the types of characteristics a person might also
# notice when deciding whether an email looks like spam.


# --- Task 4.1: Cross-Validation ---

print(
    "\nTASK 4 - CROSS VALIDATION"
)


models = {
    "KNN Unscaled":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "KNN Scaled":
        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=5
                )
            ),
        ]),

    "KNN PCA":
        Pipeline([
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
            ),
        ]),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=best_tree_depth,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "Logistic Regression Scaled":
        Pipeline([
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
            ),
        ]),

    "Logistic Regression PCA":
        Pipeline([
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
            ),
        ]),
}


cv_results = {}


for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5
    )

    cv_results[name] = {
        "mean": scores.mean(),
        "std": scores.std()
    }

    print(
        f"\n{name}"
    )

    print(
        f"Mean CV Accuracy: "
        f"{scores.mean():.4f}"
    )

    print(
        f"Standard Deviation: "
        f"{scores.std():.4f}"
    )


best_cv_model = max(
    cv_results,
    key=lambda name:
        cv_results[name]["mean"]
)

most_stable_model = min(
    cv_results,
    key=lambda name:
        cv_results[name]["std"]
)


print(
    "\nMost Accurate CV Model:",
    best_cv_model
)

print(
    "Mean Accuracy:",
    f"{cv_results[best_cv_model]['mean']:.4f}"
)


print(
    "\nMost Stable CV Model:",
    most_stable_model
)

print(
    "Standard Deviation:",
    f"{cv_results[most_stable_model]['std']:.4f}"
)


print(
    "\nBest model from single test split:",
    best_model_name
)


# The most accurate cross-validated model is the model printed above
# with the highest mean score across the five folds.
#
# The most stable model is the one with the lowest standard deviation,
# meaning its accuracy changed the least across the five folds.
#
# Comparing best_cv_model with best_model_name shows whether the model
# ranked first in cross-validation also ranked first on the original
# train/test split. A difference would show why cross-validation gives
# a more reliable comparison than relying on only one split.


# --- Task 5.1: Best Tree-Based Pipeline ---

tree_model_names = [
    "Decision Tree",
    "Random Forest"
]

best_tree_name = max(
    tree_model_names,
    key=lambda name:
        cv_results[name]["mean"]
)


non_tree_model_names = [
    "KNN Unscaled",
    "KNN Scaled",
    "KNN PCA",
    "Logistic Regression Scaled",
    "Logistic Regression PCA",
]

best_non_tree_name = max(
    non_tree_model_names,
    key=lambda name:
        cv_results[name]["mean"]
)


print(
    "\nBest Tree-Based Model:",
    best_tree_name
)

print(
    "Best Non-Tree Model:",
    best_non_tree_name
)


if best_tree_name == "Decision Tree":

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

    tree_pipeline = Pipeline([
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
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


print(
    "\nBEST TREE-BASED PIPELINE"
)

print(
    classification_report(
        y_test,
        tree_pipeline_preds
    )
)


# The tree pipeline uses the same configuration as the matching
# tree-based classifier from Task 3.
#
# It does not include StandardScaler or PCA because neither the
# Decision Tree nor Random Forest used those preprocessing steps
# in the earlier manual approach.


# --- Task 5.1: Best Non-Tree Pipeline ---

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
        ),
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
        ),
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
        ),
    ])


else:

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
        ),
    ])


non_tree_pipeline.fit(
    X_train,
    y_train
)

non_tree_pipeline_preds = (
    non_tree_pipeline.predict(
        X_test
    )
)


print(
    "\nBEST NON-TREE PIPELINE"
)

print(
    classification_report(
        y_test,
        non_tree_pipeline_preds
    )
)


# The non-tree pipeline uses exactly the preprocessing steps that
# belonged to that same model in Task 3.
#
# KNN Unscaled uses no preprocessing.
# KNN Scaled uses StandardScaler.
# KNN PCA uses StandardScaler followed by PCA.
# Logistic Regression Scaled uses StandardScaler.
# Logistic Regression PCA uses StandardScaler followed by PCA.


manual_tree_preds = prediction_lookup[
    best_tree_name
]

manual_non_tree_preds = prediction_lookup[
    best_non_tree_name
]


print(
    "\nTree Pipeline Matches Earlier Manual Predictions:",
    np.array_equal(
        tree_pipeline_preds,
        manual_tree_preds
    )
)

print(
    "Non-Tree Pipeline Matches Earlier Manual Predictions:",
    np.array_equal(
        non_tree_pipeline_preds,
        manual_non_tree_preds
    )
)


# The two True/False checks compare the pipeline predictions directly
# with the predictions produced by the same models in Task 3.
#
# A value of True confirms that packaging the preprocessing and
# classifier into a Pipeline reproduced the earlier manual approach.

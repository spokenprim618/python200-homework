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


# The cumulative explained variance first reaches 90% at 43 principal
# components.
#
# Therefore, X_train_pca and X_test_pca keep the first 43 components
# instead of all 57 original features while preserving at least 90%
# of the variance in the scaled training data.

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


# Scaled KNN achieved 0.9077 accuracy, while PCA KNN achieved 0.9066.
#
# PCA therefore reduced KNN accuracy very slightly by about 0.0011.
# The difference is extremely small, so reducing the data to 43 principal
# components preserved almost all of KNN's predictive performance, but it
# did not improve the model on this test split.
#
# Scaling itself had a much larger effect: unscaled KNN scored only
# 0.7991 compared with 0.9077 after scaling. This supports the idea that
# feature scale matters strongly for KNN because it uses distance.

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


# Training accuracy increased from 0.8965 at depth 3 to 0.9997 with
# no depth limit. Test accuracy also increased, but much more slowly:
# from 0.8849 at depth 3 to 0.9110 with no depth limit.
#
# The nearly perfect 0.9997 training accuracy for the unlimited tree
# shows that it is fitting the training data extremely closely, which
# is a sign of overfitting.
#
# I selected max_depth=None because it still produced the highest test
# accuracy, 0.9110, among the required choices. However, depth 10 was
# very close at 0.9088 while having a smaller train/test gap, so depth
# 10 would also be a reasonable more conservative production choice.

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

# Scaled Logistic Regression achieved 0.9294 accuracy, while PCA
# Logistic Regression achieved 0.9186.
#
# PCA therefore reduced Logistic Regression accuracy by about 0.0109.
# For this model, keeping all of the scaled features worked better than
# reducing them to the 43 PCA components.
#
# This suggests PCA removed some information that was still useful for
# separating spam from ham, even though those 43 components preserved
# at least 90% of the overall variance.

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


# Random Forest performed best overall on the test set with an accuracy
# of 0.9446. The next best model was scaled Logistic Regression at 0.9294.
#
# Scaling strongly improved KNN: accuracy increased from 0.7991 unscaled
# to 0.9077 scaled.
#
# PCA did not improve either of the non-tree models. KNN decreased
# slightly from 0.9077 to 0.9066 after PCA, while Logistic Regression
# decreased from 0.9294 to 0.9186.
#
# Therefore, PCA preserved most of the predictive information but did
# not improve classification performance for either KNN or Logistic
# Regression in this experiment.
#
# For a spam filter, I would not optimize accuracy alone. I would rather
# minimize false positives because a false positive sends a legitimate
# email to spam, where the user may miss something important. A false
# negative lets spam into the inbox, which is inconvenient but usually
# less costly than hiding a legitimate message.

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


# Random Forest was the best-performing classifier with test accuracy
# 0.9446, so this confusion matrix analyzes its errors specifically.
#
# It produced 18 false positives, where ham emails were incorrectly
# classified as spam, and 33 false negatives, where spam emails were
# incorrectly classified as ham.
#
# Therefore, the Random Forest makes more false negatives than false
# positives: 33 compared with 18.
#
# Even though false negatives occurred more often, I would still consider
# false positives the more costly error for this application because they
# can hide legitimate email from the user.

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

# The Decision Tree and Random Forest share 7 features in their top 10:
# capital_run_length_total, word_freq_hp, char_freq_!,
# capital_run_length_average, char_freq_$, word_freq_remove,
# and word_freq_free.
#
# This shows substantial agreement between the two tree-based models
# about which variables are useful for identifying spam.
#
# char_freq_$ is the most important Decision Tree feature, while
# char_freq_! is the most important Random Forest feature.
#
# These results match the intuition behind spam detection because
# unusual punctuation, words such as "free" and "remove", and unusual
# capitalization patterns are all characteristics that can distinguish
# spam emails from normal messages.

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

# Random Forest had the highest mean cross-validation accuracy at 0.9541,
# so it was the most accurate model across the five folds.
#
# Logistic Regression PCA had the lowest reported standard deviation at
# about 0.0077, making it the most stable model across the folds.
#
# Random Forest also had the highest accuracy on the original test split
# at 0.9446. Therefore, both the single train/test comparison and the
# cross-validation comparison ranked Random Forest first.
#
# This strengthens the evidence that Random Forest is the best classifier
# tested here because its advantage was not limited to one test split.

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


# Random Forest was the best tree-based model, so its pipeline contains
# only the RandomForestClassifier.
#
# StandardScaler and PCA are not included because the Random Forest was
# trained on the original unscaled features in Task 3. Tree-based models
# split individual features at thresholds and therefore do not require
# distance-based scaling.

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

# Scaled Logistic Regression was the best non-tree model, so this pipeline
# contains StandardScaler followed by LogisticRegression.
#
# The scaler is necessary because this is the same preprocessing used by
# the 0.9294 Logistic Regression model in Task 3.
#
# PCA is not included because PCA Logistic Regression performed worse,
# scoring 0.9186 compared with 0.9294 for the full scaled version.

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


# Both pipeline comparison checks returned True.
#
# This confirms that the Random Forest pipeline produced exactly the same
# predictions as the earlier manual Random Forest, and the scaled Logistic
# Regression pipeline produced exactly the same predictions as the earlier
# manual scaled Logistic Regression.
#
# Therefore, the pipelines successfully reproduce the preprocessing and
# classifier behavior from Task 3.
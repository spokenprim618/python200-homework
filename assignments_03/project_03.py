
import os

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline


from ucimlrepo import fetch_ucirepo 

spambase = fetch_ucirepo(id=94) 
  
X = spambase.data.features 
y = spambase.data.targets 
  
print(spambase.metadata) 
  
print(spambase.variables) 

print("DATASET OVERVIEW")

df = X.copy()

target_col = y.columns[0]

df[target_col] = y

print(f"Number of emails: {len(df)}")
print(f"Number of features (including target): {df.shape[1]}")

print("\nClass counts:")
print(df[target_col].value_counts())

print("\nClass percentages:")
print(df[target_col].value_counts(normalize=True) * 100)

spam_pct = df[target_col].mean() * 100
ham_pct = 100 - spam_pct

print(f"\nSpam: {spam_pct:.2f}%")
print(f"Ham : {ham_pct:.2f}%")



features_to_plot = [
    "word_freq_free",
    "char_freq_!",
    "capital_run_length_total",
]

for feature in features_to_plot:

    plt.figure(figsize=(6, 4))

    df.boxplot(
        column=feature,
        by=target_col,
    )

    plt.title(f"{feature} by Spam Label")
    plt.suptitle("")
    plt.xlabel("Spam Label (0=Ham, 1=Spam)")
    plt.ylabel(feature)

    plt.tight_layout()

    plt.savefig(f"outputs/{feature}_boxplot.png")
    plt.close()

print("\nSaved boxplots to outputs/")

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA()

pca.fit(X_train_scaled)

cumulative_variance = pca.explained_variance_ratio_.cumsum()

n = (cumulative_variance >= 0.90).argmax() + 1

print("\nPCA Components Needed for 90% Variance:", n)

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
)
plt.axhline(
    0.90,
    linestyle="--"
)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")
plt.tight_layout()

plt.savefig("outputs/pca_explained_variance.png")
plt.close()

X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca = pca.transform(X_test_scaled)[:, :n]

print("KNN - UNSCALED")

knn_unscaled = KNeighborsClassifier(n_neighbors=5)

knn_unscaled.fit(X_train, y_train)

preds = knn_unscaled.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, preds))

print(classification_report(y_test, preds))

print("KNN - SCALED")

knn_scaled = KNeighborsClassifier(n_neighbors=5)

knn_scaled.fit(X_train_scaled, y_train)

preds = knn_scaled.predict(X_test_scaled)

print("Accuracy:",
      accuracy_score(y_test, preds))

print(classification_report(y_test, preds))

print("KNN - PCA")

knn_pca = KNeighborsClassifier(n_neighbors=5)

knn_pca.fit(X_train_pca, y_train)

preds = knn_pca.predict(X_test_pca)

print("Accuracy:",
      accuracy_score(y_test, preds))

print(classification_report(y_test, preds))

print("DECISION TREE DEPTH COMPARISON")

depths = [3, 5, 10, None]

for depth in depths:

    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    train_acc = accuracy_score(
        y_train,
        tree.predict(X_train)
    )

    test_acc = accuracy_score(
        y_test,
        tree.predict(X_test)
    )

    print(
        f"Depth={depth} | "
        f"Train={train_acc:.4f} | "
        f"Test={test_acc:.4f}"
    )

best_tree_depth = 5

tree = DecisionTreeClassifier(
    max_depth=best_tree_depth,
    random_state=42
)

tree.fit(X_train, y_train)

tree_preds = tree.predict(X_test)

print("FINAL DECISION TREE")

print(
    "Accuracy:",
    accuracy_score(y_test, tree_preds)
)

print(classification_report(y_test, tree_preds))

rf = RandomForestClassifier(
    random_state=42
)

rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_preds)

print("RANDOM FOREST")

print("Accuracy:", rf_acc)

print(classification_report(y_test, rf_preds))

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

print("LOGISTIC REGRESSION - SCALED")

print("Accuracy:", lr_scaled_acc)

print(
    classification_report(
        y_test,
        lr_scaled_preds
    )
)

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

print("LOGISTIC REGRESSION - PCA")

print("Accuracy:", lr_pca_acc)

print(
    classification_report(
        y_test,
        lr_pca_preds
    )
)

results = {
    "Random Forest": rf_acc,
    "Logistic Regression Scaled": lr_scaled_acc,
    "Logistic Regression PCA": lr_pca_acc,
}

best_model_name = max(results, key=results.get)

print("\nBest model:", best_model_name)

if best_model_name == "Random Forest":
    best_model = rf
    best_preds = rf_preds

elif best_model_name == "Logistic Regression Scaled":
    best_model = lr_scaled
    best_preds = lr_scaled_preds

else:
    best_model = lr_pca
    best_preds = lr_pca_preds

disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_preds
)

plt.title(f"{best_model_name} Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "outputs/best_model_confusion_matrix.png"
)

plt.close()

print(
    "\nSaved confusion matrix to outputs/"
)

print("TASK 4 - CROSS VALIDATION")

models = {
    "KNN Unscaled":
        KNeighborsClassifier(n_neighbors=5),

    "KNN Scaled":
        Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ]),

    "KNN PCA":
        Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n)),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ]),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=best_tree_depth,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            random_state=42
        ),

    "Logistic Regression Scaled":
        Pipeline([
            ("scaler", StandardScaler()),
            ("classifier",
             LogisticRegression(
                 C=1.0,
                 max_iter=1000,
                 solver="liblinear"
             ))
        ]),

    "Logistic Regression PCA":
        Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n)),
            ("classifier",
             LogisticRegression(
                 C=1.0,
                 max_iter=1000,
                 solver="liblinear"
             ))
        ])
}

cv_results = {}

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    cv_results[name] = scores.mean()

    print(f"\n{name}")
    print(f"Mean Accuracy: {scores.mean():.4f}")
    print(f"Std Dev:       {scores.std():.4f}")

print("TASK 5 - PREDICTION PIPELINES")
tree_pipeline = Pipeline([
    ("classifier",
     RandomForestClassifier(
         random_state=42
     ))
])


tree_pipeline.fit(X_train, y_train)

tree_pipeline_preds = tree_pipeline.predict(X_test)


print("\nRandom Forest Pipeline")

print(
    classification_report(
        y_test,
        tree_pipeline_preds
    )
)

non_tree_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier",
     LogisticRegression(
         C=1.0,
         max_iter=1000,
         solver="liblinear"
     ))
])


non_tree_pipeline.fit(X_train, y_train)

non_tree_pipeline_preds = non_tree_pipeline.predict(X_test)


print("\nLogistic Regression Pipeline")

print(
    classification_report(
        y_test,
        non_tree_pipeline_preds
    )
)
# Question answers

# There are 4601 emails
#There is more skew in class 0
# There should be an understanding of the skew in data and lack of information could cause accuracy issues
# They both have outliers but one is farther than the rest
# There are some dramatic differences for example in outliers but in general they are similar with subtle differences
# Due to the differences of scale the model that is sensitive to this scale will become skewed and have varying results
# It is due to frequency
# Not much was needed to be done for the features because most changes will happen from the PCA process
# I see the differences made through scaled and unscaled, the differences with PCA and the varying abilities of each model. I see random forest is the easiest and most accurate
# The models with PCA did perform better and matches with expected behavior
# You would rather minimize false negatives to better reduce spam emails but marking real emails as spam could be worse
#It makes more false negatives
#Still random forest
#Logistic regression scaled
# It matches with the best model but it seems the best for variance has become more clear
#They have the same flow and the are easier to follow so more easy to maintain and for others to understand
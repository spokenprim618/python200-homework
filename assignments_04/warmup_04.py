import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
    f1_score
)
import joblib

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ROC and AUC
# ROC Q1

scaler_1 = StandardScaler()

X_train_scaled_1 = scaler_1.fit_transform(
    X_train
)

X_test_scaled_1 = scaler_1.transform(
    X_test
)


log_reg_1 = LogisticRegression(
    max_iter=1000,
    random_state=42
)

log_reg_1.fit(
    X_train,
    y_train
)


knn_clas_1 = KNeighborsClassifier(
    n_neighbors=5
)

knn_clas_1.fit(
    X_train_scaled_1,
    y_train
)


log_pred_1 = log_reg_1.predict_proba(
    X_test
)

knn_pred_1 = knn_clas_1.predict_proba(
    X_test_scaled_1
)


log_auc = roc_auc_score(
    y_test,
    log_pred_1[:, 1]
)

knn_auc = roc_auc_score(
    y_test,
    knn_pred_1[:, 1]
)


print(
    f"Logistic Regression AUC: {log_auc:.4f}"
)

print(
    f"KNN AUC: {knn_auc:.4f}"
)


# KNN is trained on scaled data because it uses distances between
# samples. Scaling prevents features with larger numerical ranges
# from having too much influence on those distances.
#
# The model with the higher AUC separates the two classes better
# across different classification thresholds.

# ROC Q2
# KNN has the fewest false positives so chose KNN for this
log_fpr_2, log_tpr_2, _ = roc_curve(y_test, log_pred_1[:, 1])
knn_fpr_2, knn_tpr_2, _ = roc_curve(y_test, knn_pred_1[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(log_fpr_2, log_tpr_2, label=f"Logistic Regression (AUC={log_auc:.3f})")
plt.plot(knn_fpr_2, knn_tpr_2, label=f"KNN (AUC={knn_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig("outputs/roc_comparison.png")
plt.close()

log_idx_2 = np.argmin(np.abs(log_tpr_2 - 0.80))
knn_idx_2 = np.argmin(np.abs(knn_tpr_2 - 0.80))

print(f"Log @ TPR≈0.80 -> FPR={log_fpr_2[log_idx_2]:.4f}, TPR={log_tpr_2[log_idx_2]:.4f}")
print(f"Knn @ TPR≈0.80 -> FPR={knn_fpr_2[knn_idx_2]:.4f}, TPR={knn_tpr_2[knn_idx_2]:.4f}")

# ROC Q3
# This increased both TPR and FPR. It would be used to be more lenient with positives but as seen also increases FPR

log_fpr_3, log_tpr_3, log_thresholds_3 = roc_curve(y_test, log_pred_1[:, 1])

best_f1_3 = -1
best_threshold_3 = None
best_tpr_3 = None
best_fpr_3 = None

for threshold, cur_tpr, cur_fpr in zip(log_thresholds_3, log_tpr_3, log_fpr_3):
    log_y_pred_3 = (log_pred_1[:, 1] >= threshold).astype(int)
    cur_f1_3 = f1_score(y_test, log_y_pred_3)

    if cur_f1_3 > best_f1_3:
        best_f1_3 = cur_f1_3
        best_threshold_3 = threshold
        best_tpr_3 = cur_tpr
        best_fpr_3 = cur_fpr

print(f"Best Threshold: {best_threshold_3:.4f}")
print(f"TPR: {best_tpr_3:.4f}")
print(f"FPR: {best_fpr_3:.4f}")
print(f"F1: {best_f1_3:.4f}")



# GridSearch

# GridSearch Q1
# I did't it picked 100 and it increased by around .06

log_reg_pipe_1 = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

log_param_grid_1 = {
    "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

log_grid_1 = GridSearchCV(
    log_reg_pipe_1,
    log_param_grid_1,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)

log_grid_1.fit(X_train,y_train)

log_reg_best_1 = log_grid_1.best_estimator_

log_test_auc_1 = roc_auc_score(
    y_test,
    log_reg_best_1.predict_proba(X_test)[:, 1],
)

print(f"Best C: {log_grid_1.best_params_['clf__C']}")
print(f"Best CV AUC: {log_grid_1.best_score_:.4f}")
print(f"Test AUC: {log_test_auc_1:.4f}")

log_reg_default_1 = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

log_reg_default_1.fit(X_train,y_train)

log_default_auc_1 = roc_auc_score(
    y_test,
    log_reg_default_1.predict_proba(X_test)[:, 1],
)



# GridSearch Q2
# The KNN CV AUC is better than logistic but not better than the test. I would take KNN farther. I would consider why with CV AUC got slightly worse and what does my goal need.

tree_clas_pipe_2 = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(random_state=42)),
    ]
)

tree_param_grid_2 = {
    "clf__max_depth": [2, 3, 5, 8, None]
}

tree_grid_2 = GridSearchCV(
    tree_clas_pipe_2,
    tree_param_grid_2,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)

tree_grid_2.fit(X_train,y_train)

tree_clas_best_2 = tree_grid_2.best_estimator_

tree_test_auc_2 = roc_auc_score(
    y_test,
    tree_clas_best_2.predict_proba(X_test)[:, 1],
)

print(f"Best max_depth: {tree_grid_2.best_params_['clf__max_depth']}")
print(f"Best CV AUC: {tree_grid_2.best_score_:.4f}")
print(f"Test AUC: {tree_test_auc_2:.4f}")



# GridSearch Q3
# I would pick the mean with the smallest STD as it would make my model less impacted by variance

log_results_3 = pd.DataFrame(log_grid_1.cv_results_)

log_summary_3 = log_results_3[
    [
        "param_clf__C",
        "mean_test_score",
        "std_test_score",
    ]
].sort_values("mean_test_score", ascending=False)

print(log_summary_3.to_string(index=False))



# joblib

# joblib Q1
# THe model will be impacted by the varience of not being scaled making it unreliable

joblib.dump(log_reg_best_1, "models/warmup_model.pkl")

loaded_log_reg_1 = joblib.load("models/warmup_model.pkl")

log_preds_orig_1 = log_reg_best_1.predict(X_test)
log_preds_loaded_1 = loaded_log_reg_1.predict(X_test)

assert (log_preds_orig_1 == log_preds_loaded_1).all(), "Predictions do not match!"

print("Predictions match. Model saved and loaded successfully.")


# joblib Q2
# I expect to predict the probability to be somewhere in the middle because of the pos and neg numbers around it

loaded_log_reg_2 = joblib.load("models/warmup_model.pkl")

 # --- Simulated prediction script ---

new_samples_2 = np.array(
    [
        [2.5, 1.2, -0.3, 0.8, 1.0, -0.5, 0.2, 0.9, -1.1, 0.4],
        [-1.0, 0.5, 0.9, -0.7, -0.2, 1.3, -0.8, 0.1, 0.5, -0.3],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)

log_preds_2 = loaded_log_reg_2.predict(new_samples_2)
log_probs_2 = loaded_log_reg_2.predict_proba(new_samples_2)[:, 1]

for i, (pred, prob) in enumerate(zip(log_preds_2, log_probs_2), start=1):
    print(f"Sample {i}: class={pred}, probability={prob:.4f}")

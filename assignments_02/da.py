"""
Student Performance Analysis
----------------------------

Research Questions:

1. Which variables are most correlated with G3?
2. Which predictors exhibit multicollinearity?
3. Which features add unique predictive value?
4. Can social behavior be represented by a combined social_risk feature?
   """

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold

from statsmodels.stats.outliers_influence import variance_inflation_factor

# ==========================================================

# CONFIGURATION

# ==========================================================

DATA_PATH = "./data/student_performance_math.csv"

TARGET = "G3"

CATEGORICAL_COLUMNS = [
"sex",
"schoolsup",
"internet",
"higher",
"activities"
]

# ==========================================================

# DATA LOADING / PREPROCESSING

# ==========================================================

def load_data():
"""Load raw dataset."""
return pd.read_csv(DATA_PATH, sep=";")

def encode_data(df):
"""One-hot encode categorical variables."""
return pd.get_dummies(
df,
columns=CATEGORICAL_COLUMNS,
drop_first=True
)

# ==========================================================

# CORRELATION ANALYSIS

# ==========================================================

def show_target_correlations(df, target=TARGET):
"""Show correlations with target variable."""

```
correlations = (
    df.corr(numeric_only=True)[target]
    .sort_values(key=abs, ascending=False)
)

print("\n=== CORRELATIONS WITH G3 ===")
print(correlations)

return correlations
```

def correlation_heatmap(df, predictors):
"""Visualize predictor correlations."""

```
corr_matrix = df[predictors].corr()

print("\n=== PREDICTOR CORRELATIONS ===")
print(corr_matrix)

plt.figure(figsize=(8, 6))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    center=0
)

plt.title("Predictor Correlation Matrix")
plt.show()

return corr_matrix
```

# ==========================================================

# MULTICOLLINEARITY ANALYSIS

# ==========================================================

def calculate_vif(df, predictors):
"""Calculate variance inflation factors."""

```
X = df[predictors].astype(float)

vif_df = pd.DataFrame({
    "feature": X.columns,
    "VIF": [
        variance_inflation_factor(
            X.values,
            i
        )
        for i in range(X.shape[1])
    ]
})

vif_df = vif_df.sort_values(
    "VIF",
    ascending=False
)

print("\n=== VIF ANALYSIS ===")
print(vif_df)

return vif_df
```

# ==========================================================

# FEATURE SELECTION

# ==========================================================

def run_rfecv(df, target=TARGET):
"""Perform recursive feature elimination."""

```
X = df.drop(columns=[target])
y = df[target]

rfecv = RFECV(
    estimator=LinearRegression(),
    step=1,
    cv=KFold(
        5,
        shuffle=True,
        random_state=42
    ),
    scoring="r2"
)

rfecv.fit(X, y)

print("\n=== RFECV RESULTS ===")

print(
    "Optimal number of features:",
    rfecv.n_features_
)

selected_features = X.columns[
    rfecv.support_
]

print("\nSelected Features:")
print(selected_features)

rankings = pd.DataFrame({
    "feature": X.columns,
    "rank": rfecv.ranking_
}).sort_values("rank")

print("\nFeature Rankings:")
print(rankings)

return rfecv, rankings
```

def plot_rfecv_results(rfecv):
"""Plot RFECV performance curve."""

```
plt.figure(figsize=(8, 5))

plt.plot(
    range(
        1,
        len(
            rfecv.cv_results_[
                "mean_test_score"
            ]
        ) + 1
    ),
    rfecv.cv_results_[
        "mean_test_score"
    ]
)

plt.xlabel("Number of Features")
plt.ylabel("Cross-Validated R²")
plt.title("RFECV Feature Selection")

plt.grid(True)

plt.show()
```

# ==========================================================

# SOCIAL RISK ANALYSIS

# ==========================================================

def create_social_features(df):
"""Create combined social behavior features."""

```
df = df.copy()

df["social_sum"] = (
    df["goout"] + df["Walc"]
)

df["social_avg"] = (
    df["goout"] + df["Walc"]
) / 2

df["social_interaction"] = (
    df["goout"] * df["Walc"]
)

return df
```

def analyze_social_features(df):
"""Compare social variables against G3."""

```
print("\n=== SOCIAL FEATURE ANALYSIS ===")

variables = [
    "goout",
    "Walc",
    "social_sum",
    "social_avg",
    "social_interaction"
]

for col in variables:

    print(
        f"{col:20}",
        round(
            df[col].corr(df["G3"]),
            4
        )
    )
```

# ==========================================================

# GROUP COMPARISON ANALYSIS

# ==========================================================

def compare_group_means(df, column):
"""Compare average G3 by category."""

```
print(
    f"\n=== G3 BY {column.upper()} ==="
)

print(
    df.groupby(column)["G3"]
    .mean()
    .sort_values(
        ascending=False
    )
)
```

# ==========================================================

# MAIN ANALYSIS PIPELINE

# ==========================================================

def main():

```
# --------------------------------------
# Load and prepare data
# --------------------------------------

df = load_data()

df_encoded = encode_data(df)

# --------------------------------------
# Correlation analysis
# --------------------------------------

show_target_correlations(
    df_encoded
)

correlation_heatmap(
    df_encoded,
    [
        "absences",
        "failures",
        "goout",
        "Walc"
    ]
)

# --------------------------------------
# Multicollinearity
# --------------------------------------

calculate_vif(
    df_encoded,
    [
        "absences",
        "failures",
        "goout",
        "Walc"
    ]
)

# --------------------------------------
# Feature selection
# --------------------------------------

rfecv, rankings = run_rfecv(
    df_encoded
)

plot_rfecv_results(
    rfecv
)

# --------------------------------------
# Social behavior analysis
# --------------------------------------

df_social = create_social_features(
    df
)

analyze_social_features(
    df_social
)

# --------------------------------------
# Future investigations
# --------------------------------------

compare_group_means(
    df,
    "schoolsup"
)

compare_group_means(
    df,
    "activities"
)
```

if **name** == "**main**":
main()

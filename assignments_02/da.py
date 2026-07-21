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

    correlations = (
        df.corr(numeric_only=True)[target]
        .sort_values(key=abs, ascending=False)
    )

    print("\n=== CORRELATIONS WITH G3 ===")
    print(correlations)

    return correlations


def correlation_heatmap(df, predictors):
    """Visualize predictor correlations."""

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


# ==========================================================
# MULTICOLLINEARITY ANALYSIS
# ==========================================================

def calculate_vif(df, predictors):
    """Calculate variance inflation factors."""

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

# ==========================================================
# FEATURE SELECTION
# ==========================================================

def run_rfecv(df, target=TARGET):
    """Perform recursive feature elimination."""

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


def plot_rfecv_results(rfecv):
    """Plot RFECV performance curve."""

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


# ==========================================================
# SOCIAL RISK ANALYSIS
# ==========================================================

def create_social_features(df):
    """Create combined social behavior features."""

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


def analyze_social_features(df):
    """Compare social variables against G3."""

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


# ==========================================================
# GROUP COMPARISON ANALYSIS
# ==========================================================

def compare_group_means(df, column):
    """Compare average G3 by category."""

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

# ==========================================================
# FEATURE GROUPING / COMPOSITE FEATURES
# ==========================================================

def create_group_features(df):
    """
    Create combined features representing underlying factors.
    """

    df = df.copy()

    # Academic performance factor
    df["academic_score"] = (
        df["G1"]
        + df["G2"]
        + df["studytime"]
        - df["failures"]
    )


    # Educational environment factor
    df["education_environment"] = (
        df["Medu"]
        + df["Fedu"]
        + df["higher_yes"]
    ) / 3


    # Student resources/support factor
    df["student_resources"] = (
        df["schoolsup_yes"]
        + df["activities_yes"]
        + df["internet_yes"]
    ) / 3


    # Social behavior factor
    df["social_behavior"] = (
        df["goout"]
        + df["Walc"]
        + df["freetime"]
    ) / 3


    return df


def analyze_group_features(df):
    """
    Analyze correlations of combined features with G3.
    """

    print("\n=== GROUP FEATURE ANALYSIS ===")

    features = [
        "academic_score",
        "education_environment",
        "student_resources",
        "social_behavior"
    ]

    for feature in features:

        print(
            f"{feature:25}",
            round(
                df[feature].corr(df["G3"]),
                4
            )
        )


def compare_group_models(df):
    """
    Compare grouped features against original features.
    """

    grouped_features = [
        "academic_score",
        "education_environment",
        "student_resources",
        "social_behavior"
    ]

    original_features = [
        "G1",
        "G2",
        "failures",
        "studytime",
        "Medu",
        "Fedu",
        "schoolsup_yes",
        "activities_yes",
        "Walc",
        "goout"
    ]

    print("\n=== GROUPED FEATURES ===")
    print(grouped_features)

    print("\n=== ORIGINAL FEATURES ===")
    print(original_features)

def analyze_group_contributions(df, features):
    """
    Show contribution of each feature to a combined score.
    """

    print("\n=== GROUP CONTRIBUTIONS ===")

    group_df = df[features].copy()

    standardized = (
        group_df - group_df.mean()
    ) / group_df.std()

    contribution = (
        standardized.abs()
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print(contribution)

    return contribution
def create_correlation_groups(
    df,
    target_size=3,
    threshold=0.4
):
    """
    Create groups of correlated features.

    target_size:
        Number of variables per group

    threshold:
        Minimum absolute correlation
    """

    corr_matrix = df.corr(
        numeric_only=True
    )

    features = [
        col for col in corr_matrix.columns
        if col != TARGET
    ]

    groups = []

    used = set()


    for feature in features:

        if feature in used:
            continue


        correlations = (
            corr_matrix[feature]
            .drop(feature)
            .abs()
            .sort_values(
                ascending=False
            )
        )


        selected = [
            feature
        ]


        for other, corr in correlations.items():

            if (
                corr >= threshold
                and other not in used
            ):
                selected.append(other)

            if len(selected) == target_size:
                break


        if len(selected) == target_size:

            groups.append(selected)

            used.update(selected)


    return groups
def create_features_from_groups(df, groups):

    df = df.copy()

    for i, group in enumerate(groups):

        name = (
            f"group_feature_{i+1}"
        )

        df[name] = (
            df[group]
            .mean(axis=1)
        )


    return df
def create_dynamic_correlation_groups(
    df,
    start_threshold=0.4,
    decrement=0.05,
    min_threshold=0.1
):
    """
    Dynamically create correlation groups.

    Starts at start_threshold.
    Lowers threshold when no groups are found.

    Group expansion uses correlation between
    the combined group feature and new variables.
    """

    remaining_features = [
        col for col in df.columns
        if col != TARGET
    ]

    groups = []

    threshold = start_threshold


    while remaining_features:


        # Stop condition

        if len(remaining_features) == 1:

            groups.append(
                remaining_features
            )

            break


        corr_matrix = (
            df[remaining_features]
            .corr()
            .abs()
        )


        best_pair = None
        best_corr = 0


        # Find strongest starting pair

        for i, feature1 in enumerate(
            remaining_features
        ):

            for feature2 in remaining_features[i+1:]:

                corr = corr_matrix.loc[
                    feature1,
                    feature2
                ]


                if (
                    corr >= threshold
                    and corr > best_corr
                ):

                    best_corr = corr

                    best_pair = [
                        feature1,
                        feature2
                    ]


        # If no pair exists lower threshold

        if best_pair is None:

            threshold -= decrement


            if threshold < min_threshold:

                for feature in remaining_features:

                    groups.append(
                        [feature]
                    )

                break


            continue



        # Create group

        group = best_pair.copy()


        remaining_features = [
            f for f in remaining_features
            if f not in group
        ]


        # Expand group

        expanded = True

        while expanded:

            expanded = False


            group_feature = (
                df[group]
                .mean(axis=1)
            )


            best_candidate = None
            best_candidate_corr = 0


            for feature in remaining_features:

                corr = abs(
                    group_feature.corr(
                        df[feature]
                    )
                )


                if (
                    corr >= threshold
                    and corr > best_candidate_corr
                ):

                    best_candidate_corr = corr

                    best_candidate = feature


            if best_candidate is not None:

                group.append(
                    best_candidate
                )

                remaining_features.remove(
                    best_candidate
                )

                expanded = True



        groups.append(
            group
        )


    return groups
    
def group_strength(df, group):

    # Single feature groups have no internal correlation
    if len(group) < 2:
        return None


    corr = (
        df[group]
        .corr()
        .abs()
    )

    values = []


    for i in range(len(group)):

        for j in range(i + 1, len(group)):

            values.append(
                corr.iloc[i, j]
            )


    if len(values) == 0:
        return None


    return round(
        sum(values) / len(values),
        3
    )

# ==========================================================
# GROUP TARGET ANALYSIS
# ==========================================================

def analyze_group_target_correlations(df, groups):
    """
    Check correlation of each group with target.
    """

    print("\n=== INDIVIDUAL GROUP TARGET CORRELATIONS ===")


    results = []


    for i, group in enumerate(groups, 1):

        group_feature = (
            df[group]
            .mean(axis=1)
        )


        correlation = (
            group_feature
            .corr(df[TARGET])
        )


        print(
            f"Group {i}: {group}"
        )

        print(
            "Correlation:",
            round(
                correlation,
                4
            )
        )

        results.append(
            {
                "group": i,
                "features": group,
                "correlation": correlation
            }
        )


    return pd.DataFrame(results)



def create_group_dataframe(df, groups):
    """
    Create dataframe containing only
    automatically generated group features.
    """

    group_df = pd.DataFrame()


    for i, group in enumerate(groups, 1):

        group_df[
            f"group_{i}"
        ] = (
            df[group]
            .mean(axis=1)
        )


    group_df[TARGET] = df[TARGET]


    return group_df
# ==========================================================
# MAIN ANALYSIS PIPELINE
# ==========================================================

def main():

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


        # ==========================================================
    # FUTURE INVESTIGATIONS / FEATURE ENGINEERING
    # ==========================================================


    # --------------------------------------
    # Group Mean Comparisons
    # --------------------------------------

    compare_group_means(
        df,
        "schoolsup"
    )

    compare_group_means(
        df,
        "activities"
    )



    # --------------------------------------
    # Manual Composite Feature Analysis
    # --------------------------------------
    # Tests manually created latent features:
    # academic_score
    # education_environment
    # student_resources
    # social_behavior

    df_grouped = create_group_features(
        df_encoded
    )

    analyze_group_features(
        df_grouped
    )

    compare_group_models(
        df_grouped
    )



    # --------------------------------------
    # Individual Variable Contribution
    # --------------------------------------
    # Shows which variables contribute most
    # to a combined feature

    analyze_group_contributions(
        df_encoded,
        [
            "goout",
            "Walc",
            "freetime"
        ]
    )



    # --------------------------------------
    # Automatic Correlation-Based Grouping
    # --------------------------------------
    # Finds groups of similar variables
    # based on correlation threshold

    groups = create_correlation_groups(
        df_encoded,
        target_size=3,
        threshold=0.4
    )

    print("\n=== AUTOMATIC CORRELATION GROUPS ===")

    for i, group in enumerate(groups, 1):

        print(
            f"Group {i}:",
            group
        )



    # --------------------------------------
    # Create Features From Correlation Groups
    # --------------------------------------

    df_auto_grouped = create_features_from_groups(
        df_encoded,
        groups
    )

    print("\n=== GENERATED GROUP FEATURES ===")

    print(
        df_auto_grouped[
            [
                col for col in df_auto_grouped.columns
                if "group_feature" in col
            ]
        ].head()
    )
    groups = create_dynamic_correlation_groups(
    df_encoded
)


    print("\n=== RECURSIVE CORRELATION GROUPS ===")


    for i, group in enumerate(groups, 1):

        print(
            f"Group {i}: {group}"
        )
    for i, group in enumerate(groups, 1):

        print(
            f"\nGroup {i}: {group}"
        )

        strength = group_strength(
            df_encoded,
            group
        )


        if strength is not None:

            print(
                "Strength:",
                strength
            )

        else:

            print(
                "Strength: Single feature group"
            )
    groups = create_dynamic_correlation_groups(
        df_encoded
    )
        # --------------------------------------
    # Analyze Individual Groups
    # --------------------------------------

    group_results = analyze_group_target_correlations(
        df_encoded,
        groups
    )


    print(group_results)



    # --------------------------------------
    # Analyze All Groups Together
    # --------------------------------------

    df_group_analysis = create_group_dataframe(
        df_encoded,
        groups
    )


    print(
        "\n=== GROUP FEATURE CORRELATIONS ==="
    )


    show_target_correlations(
        df_group_analysis
    )



    print(
        "\n=== GROUP FEATURE VIF ==="
    )


    calculate_vif(
        df_group_analysis,
        [
            col for col in df_group_analysis.columns
            if col != TARGET
        ]
    )



    print(
        "\n=== GROUP FEATURE RFECV ==="
    )


    rfecv_groups, rankings_groups = run_rfecv(
        df_group_analysis
    )


    plot_rfecv_results(
        rfecv_groups
    )
if __name__ == "__main__":
    main()
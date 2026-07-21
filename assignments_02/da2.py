from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np

DATA_PATH = "./data/student_performance_math.csv"

TARGET = "G3"

CATEGORICAL_COLUMNS = [
    "sex",
    "schoolsup",
    "internet",
    "higher",
    "activities"
]



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

df = load_data()

df_encoded = encode_data(df)


def evaluate_model(df, features, target="G3"):

    X = df[features]
    y = df[target]

    model = LinearRegression()

    r2_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="r2"
    )

    rmse_scores = np.sqrt(
        -cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring="neg_mean_squared_error"
        )
    )

    return {
        "features": features,
        "R2 Mean": round(r2_scores.mean(), 4),
        "RMSE Mean": round(rmse_scores.mean(), 4)
    }


base_features = [
    "age",
    "studytime",
    "failures",
    "absences"
]


models = [

    base_features,

    base_features + [
        "goout"
    ],

    base_features + [
        "Walc"
    ],

    base_features + [
        "goout",
        "Walc"
    ],

    base_features + [
        "goout",
        "Walc",
        "freetime",
        "activities_yes"
    ],

    base_features + [
        "goout",
        "Walc",
        "freetime",
        "activities_yes",
        "schoolsup_yes"
    ]

]


results = []

for features in models:

    results.append(
        evaluate_model(
            df_encoded,
            features
        )
    )


results_df = pd.DataFrame(results)

results_df = pd.DataFrame(results)

results_df.to_csv(
    "./outputs/nested_model_results.csv",
    index=False
)


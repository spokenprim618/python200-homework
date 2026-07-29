# Step 1
import json
import platform

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import requests
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LATITUDE = 25.7617
LONGITUDE = -80.1918

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}

response = requests.get(url, params=params)
response.raise_for_status()


df = pd.DataFrame(response.json()["daily"])

df["date"] = pd.to_datetime(df["time"])
df = df.drop(columns=["time"])

print(df.head())
print(df.info())
print(df.describe())

# Step 2
# I informed my decision based on these values in florida to what could be preferable. 0.32 makes sense as shown because of all the rain and extreme weather days.

print("Max temp condition:")
print(df["temperature_2m_max"].between(20, 30).value_counts())

print("Min temp condition:")
print((df["temperature_2m_min"] >= 15).value_counts())

print("Precip condition:")
print((df["precipitation_sum"] < 3.0).value_counts())

print("Wind condition:")
print((df["wind_speed_10m_max"] < 25).value_counts())


print(response.json()["daily_units"])

print(df["temperature_2m_max"].describe())
print(df["temperature_2m_min"].describe())



df["good_for_running"] = (
    df["temperature_2m_max"].between(20, 30)
    & (df["temperature_2m_min"] >= 15)
    & (df["precipitation_sum"] < 3.0)
    & (df["wind_speed_10m_max"] < 25)
).astype(int)

print(df["good_for_running"].value_counts())
print(df["good_for_running"].value_counts(normalize=True))
features = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

X = df[features]
y = df["good_for_running"]

# Step 3


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ]
)

param_grid = {
    "clf__C": [
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    ]
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)
y_probs = best_model.predict_proba(X_test)[:, 1]

test_auc = roc_auc_score(y_test, y_probs)

print(f"Best C: {grid.best_params_['clf__C']}")
print(f"Best CV AUC: {grid.best_score_:.4f}")

print(classification_report(y_test, y_pred))

print(f"Test AUC: {test_auc:.4f}")

fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {test_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Weather Classifier ROC Curve")
plt.legend()
plt.savefig("outputs/weather_roc.png")
plt.close()

# Step 4
# Both the test and CV logistic models do a good job at seperating the classes. I am suprised the model was good at determining and should be looked into more.
# It seems false positives are more common due precision for class 1 being 0.76 which is the lowest.
# I would rather the app under recommends because someone could be caught in rain or bad termperature for their purposes
# Keep at test because it did the best even if slightly


#Step 5


joblib.dump(
    best_model,
    "models/weather_classifier.pkl",
)

metadata = {
    "python_version": platform.python_version(),
    "scikit_learn_version": sklearn.__version__,
    "feature_names": features,
    "best_hyperparameters": grid.best_params_,
    "test_auc": test_auc,
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "label_thresholds": {
        "temperature_2m_max": "20 <= x <= 30",
        "temperature_2m_min": "x >= 15",
        "precipitation_sum": "x < 3.0",
        "wind_speed_10m_max": "x < 25",
    },
}

with open(
    "models/weather_classifier_metadata.json",
    "w",
) as f:
    json.dump(metadata, f, indent=4)

print("Saved models/weather_classifier.pkl")
print("Saved models/weather_classifier_metadata.json")
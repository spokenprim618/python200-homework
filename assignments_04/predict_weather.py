# Task 1

import json

import joblib
import pandas as pd

model = joblib.load("models/weather_classifier.pkl")

with open(
    "models/weather_classifier_metadata.json",
    "r",
) as f:
    metadata = json.load(f)

print(f"Latitude: {metadata['latitude']}")
print(f"Longitude: {metadata['longitude']}")
print(f"Features: {metadata['feature_names']}")
print(f"Test AUC: {metadata['test_auc']:.4f}")

feature_names = metadata["feature_names"]

# Task 2

new_days = pd.DataFrame(
    [
        [24, 18, 0.0, 10],
        [28, 20, 1.0, 18],
        [31, 16, 2.5, 24],
        [22, 18, 12.0, 15],
        [25, 17, 0.0, 45],
        [8, 2, 0.0, 12],
    ],
    columns=feature_names,
)

probs = model.predict_proba(new_days)[:, 1]
preds = model.predict(new_days)

for i in range(len(new_days)):
    row = new_days.iloc[i]

    label = "good" if preds[i] == 1 else "skip"

    print(f"\nDay {i + 1}")
    print(f"temperature_2m_max: {row['temperature_2m_max']}")
    print(f"temperature_2m_min: {row['temperature_2m_min']}")
    print(f"precipitation_sum: {row['precipitation_sum']}")
    print(f"wind_speed_10m_max: {row['wind_speed_10m_max']}")
    print(f"Prediction: {label}")
    print(f"Probability Good For Running: {probs[i]:.4f}")

# Task 3

# The probability for borderline was 0.0062. It seems very confident no. I would want the model to still state not to run to be safe
# The model would not be found and this should be stated so the user understands the order of operations in the error or have a main runner file to avoid this
# Instead of making fake days, real current days will be fed to be classified by the model
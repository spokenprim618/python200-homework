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

day_names = [
    "Clearly Good Day 1",
    "Clearly Good Day 2",
    "Borderline Day",
    "Rainy Day",
    "Very Windy Day",
    "Cold Day",
]

probs = model.predict_proba(new_days)[:, 1]
preds = model.predict(new_days)

for i in range(len(new_days)):
    row = new_days.iloc[i]

    label = "good" if preds[i] == 1 else "skip"

    print(
        f"\nDay {i + 1}: "
        f"{day_names[i]}"
    )

    print(f"temperature_2m_max: {row['temperature_2m_max']}")
    print(f"temperature_2m_min: {row['temperature_2m_min']}")
    print(f"precipitation_sum: {row['precipitation_sum']}")
    print(f"wind_speed_10m_max: {row['wind_speed_10m_max']}")
    print(f"Prediction: {label}")
    print(f"Probability Good For Running: {probs[i]:.4f}")

# Task 3

# Q1
# I used Day 3 as my borderline case. Its predicted probability was about
# 0.0062 in my run, so despite designing the weather values to be near some
# of my label boundaries, the model itself was very confident that this
# should be a "skip" day.
#
# A probability such as 0.52 would be very different because it is only
# slightly above the default 0.50 threshold. I would describe that result
# as uncertain rather than strongly recommending a run. In a real app I
# would probably communicate that uncertainty to the user or use a higher
# recommendation threshold if I wanted to avoid unsafe recommendations.


# Q2
# If predict_weather.py is run before train_weather_classifier.py, the
# model and metadata files may not exist yet. joblib.load() or open() would
# then raise a file-not-found error before any prediction could be made.
#
# I could make this more helpful by checking whether both files exist first
# and raising a clear message such as: "Weather model not found. Run
# train_weather_classifier.py before predict_weather.py." A main runner
# script could also make sure training happens before prediction when a
# model has not been created yet.


# Q3
# For daily production use, I would replace the manually constructed
# hypothetical days with tomorrow's real weather forecast from a weather
# API. The prediction script would need to request the same four features
# used during training and put them into a DataFrame with exactly the same
# feature names and order stored in the metadata.
#
# The trained Pipeline could then predict tomorrow's class and probability
# without retraining the model every day. The script could also be
# scheduled to run automatically each day and send or display the resulting
# recommendation to the user.

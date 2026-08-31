import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os

file = "data\\student_performance_math.csv"

# Task 1
# The seperator is ;
df = pd.read_csv(file, sep=";")

print("Shape:", df.shape)
print("\nFirst Five Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

plt.figure(figsize=(8, 5))
plt.hist(df["G3"], bins=21)
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Count")
plt.savefig("outputs\\g3_distribution.png")
plt.close()

# Task 2

print("\nOriginal Shape:", df.shape)

df_clean = df[df["G3"] != 0].copy()
# Keeping these rows of math grades = 0 would distort due to the large gap of 0 to other grades, there could be someone who didn't take the exam which can't be shown as predicted the model would think this is someone who after all the weights got a zero not that they never showed up as an example
# 0 also needed to be removed because it gives the wrong idea for the model to see such a sharp decrease and the model's conclusion could be off and create wrong correlations
print("Filtered Shape:", df_clean.shape)

binary_cols = ["schoolsup","internet","higher","activities"]

for col in binary_cols:
    df_clean[col] = df_clean[col].map({"yes": 1, "no": 0})

df_clean["sex"] = df_clean["sex"].map({"F": 0, "M": 1})

df_temp = df.copy()

for col in binary_cols:
    df_temp[col] = df_temp[col].map({"yes": 1, "no": 0})

df_temp["sex"] = df_temp["sex"].map({"F": 0, "M": 1})

corr_original, _ = pearsonr(df_temp["absences"], df_temp["G3"])
corr_filtered, _ = pearsonr(df_clean["absences"], df_clean["G3"])

print("\nAbsences vs G3 Correlation (Original):", corr_original)
print("Absences vs G3 Correlation (Filtered):", corr_filtered)

# Maybe because the students dropped out because a 0 in G3 also had a 0 in G2 so there wouldnt be enough information for abscences because they never had the abcenses they would have if they stayed

plt.figure(figsize=(8, 5))
plt.scatter(df_temp["absences"], df_temp["G3"], alpha=0.6)
plt.title("Absences vs G3 (Original Data)")
plt.xlabel("Absences")
plt.ylabel("G3")
plt.savefig("outputs\\absences_vs_g3_original.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(df_clean["absences"], df_clean["G3"], alpha=0.6)
plt.title("Absences vs G3 (Filtered Data)")
plt.xlabel("Absences")
plt.ylabel("G3")
plt.savefig("outputs\\absences_vs_g3_filtered.png")
plt.close()


# Task 3

numeric_df = df_clean.select_dtypes(include=[np.number])

corrs = numeric_df.corr()["G3"].sort_values()

print("\nCorrelations with G3:")
print(corrs)

plt.figure(figsize=(8, 5))
plt.scatter(df_clean["failures"], df_clean["G3"], alpha=0.6)
plt.title("Failures vs Final Grade")
plt.xlabel("Failures")
plt.ylabel("G3")
plt.savefig("outputs\\failures_vs_g3.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(df_clean["studytime"], df_clean["G3"], alpha=0.6)
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("G3")
plt.savefig("outputs\\studytime_vs_g3.png")
plt.close()

# Well the predictor being so correlated to itself isnt and previous grade and family education isnt suprising either
# Failures seem to be the best because atleast there is a likelyhood that if someone previously failed more times they would get a lower grade compared time studying seems to not matter

# Task 4

X = df_clean[["failures"]].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)

y_pred = baseline_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nBaseline Model")
print("Slope:", baseline_model.coef_[0])
print("RMSE:", rmse)
print("R²:", r2)

# The R^2 is lower than expected there is a lot of errors. Slope says as failure increases score decreases which makes sense. RMSE means there is about an error of 3 grade points which is large in a small range

# Task 5

feature_cols = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                "absences", "freetime", "goout", "Walc", "schoolsup",
                "internet", "higher", "activities", "sex"]
X = df_clean[feature_cols].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

train_r2 = model.score(X_train, y_train)

y_pred = model.predict(X_test)

test_r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nFull Model")
print("Train R²:", train_r2)
print("Test R²:", test_r2)
print("RMSE:", rmse)

print("\nCoefficients")

for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# It did improve the RMSE but it increased both training and test R^2. THis tells me the model needed more data or more accurate features to make better predictions.
# I was suprised with how strong schoolsup and internet are for correlation. I would drop those with the lowest correlation such as activities or travel time because they dont contribute as much as the higher correlated features.

# Task 6

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, y_test, alpha=0.6)

min_val = min(y_pred.min(), y_test.min())
max_val = max(y_pred.max(), y_test.max())

plt.plot([min_val, max_val], [min_val, max_val])

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")

plt.savefig("outputs\\predicted_vs_actual.png")
plt.close()

print("\nSummary")
print("Filtered Dataset Size:", len(df_clean))
print("Test Set Size:", len(y_test))
print("Best Model RMSE:", rmse)
print("Best Model R²:", test_r2)

feature_cols_g1 = [
    "failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
    "internet", "sex", "freetime", "activities", "traveltime", "G1"
]

X = df_clean[feature_cols_g1].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

g1_model = LinearRegression()
g1_model.fit(X_train, y_train)

y_pred = g1_model.predict(X_test)

g1_r2 = r2_score(y_test, y_pred)

print("\nModel Including G1")
print("Test R²:", g1_r2)

# The values are all clustered in the middle then spreading left and right. so many of similiar error in this region guessing above or below .
# Values above are over estimated and values above are under estimated. 

# The filtered dataset size is 357 and the test set is 72.
# The RMSE within the range acceptable but can be large for the small range. The R^2 is low and can be seen how many errors there were in predictions
# Typical predition error just gives you an idea of performance
# largest pos was internet and largest neg was schoolsup this means they had the strongest correlation to G3 most related
# I was suprised with how little freetime had an impact on G3

# I don't think G1 causes G3 it is most likely an indicator of a greater problem however that needs more context
# It could be because it shows it may be likely G1 scores show habits that will continue till G3 the person may not change without intervention if it is in a bad direction
# Seeing the correlations of the smaller sections such as time studying or getting help or seeing the persons background such as parental education


# I really don't know why but it was something with the additional features maybe it no longer became so important like more important features came in
# Maybe along with abscenses the want for education didn't matter but going out more did. Going out more could be a higher correlated due to the behavriol aspects 
# These are both habitual and most times are addicitve which can hurt scores
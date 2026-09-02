import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# --- Setup ---

file = os.path.join("data", "student_performance_math.csv")
os.makedirs("outputs", exist_ok=True)


# --- Task 1 ---

df = pd.read_csv(file, sep=";")

print("Shape:", df.shape)

print("\nFirst Five Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

plt.figure(figsize=(8, 5))
plt.hist(df["G3"], bins=21)
plt.title("Distribution of Student Final Math Grades (G3)")
plt.xlabel("Final Math Grade (0-20)")
plt.ylabel("Number of Students")
plt.savefig(os.path.join("outputs", "g3_distribution.png"))
plt.close()


# --- Task 2 ---

print("\nOriginal Shape:", df.shape)

df_clean = df[df["G3"] != 0].copy()

# G3 = 0 represents students who were absent from the final exam rather than
# students who actually earned a true final grade of zero.
# Keeping these rows could make the model and correlations treat an absent exam
# as extremely poor academic performance, which would distort the results.

print("Filtered Shape:", df_clean.shape)

binary_cols = ["schoolsup", "internet", "higher", "activities"]

for col in binary_cols:
    df_clean[col] = df_clean[col].map({"yes": 1, "no": 0})

df_clean["sex"] = df_clean["sex"].map({"F": 0, "M": 1})


# Create another version for checking the original correlation.
df_temp = df.copy()

for col in binary_cols:
    df_temp[col] = df_temp[col].map({"yes": 1, "no": 0})

df_temp["sex"] = df_temp["sex"].map({"F": 0, "M": 1})


corr_original, _ = pearsonr(df_temp["absences"], df_temp["G3"])
corr_filtered, _ = pearsonr(df_clean["absences"], df_clean["G3"])

print("\nAbsences vs G3 Correlation (Original):", corr_original)
print("Absences vs G3 Correlation (Filtered):", corr_filtered)

# The correlation changes because the original dataset includes students with
# G3 = 0 who were absent from the final exam. Their zeros do not represent
# normal academic performance, so they can change the relationship between
# absences and final grades. Filtering them gives a better picture of the
# relationship among students who actually received a final grade.


plt.figure(figsize=(8, 5))
plt.scatter(df_temp["absences"], df_temp["G3"], alpha=0.6)
plt.title("Student Absences vs Final Math Grade - Original Data")
plt.xlabel("Number of Absences")
plt.ylabel("Final Math Grade (G3)")
plt.savefig(os.path.join("outputs", "absences_vs_g3_original.png"))
plt.close()


plt.figure(figsize=(8, 5))
plt.scatter(df_clean["absences"], df_clean["G3"], alpha=0.6)
plt.title("Student Absences vs Final Math Grade - Filtered Data")
plt.xlabel("Number of Absences")
plt.ylabel("Final Math Grade (G3)")
plt.savefig(os.path.join("outputs", "absences_vs_g3_filtered.png"))
plt.close()


# --- Task 3 ---

numeric_df = df_clean.select_dtypes(include=[np.number])

# sort_values() sorts from the most negative correlation
# to the most positive correlation.
corrs = numeric_df.corr()["G3"].sort_values()

print("\nCorrelations with G3 (Most Negative to Most Positive):")
print(corrs)


plt.figure(figsize=(8, 5))
plt.scatter(df_clean["failures"], df_clean["G3"], alpha=0.6)
plt.title("Previous Class Failures vs Final Math Grade")
plt.xlabel("Number of Previous Class Failures")
plt.ylabel("Final Math Grade (G3)")
plt.savefig(os.path.join("outputs", "failures_vs_g3.png"))
plt.close()

# This plot shows a negative relationship. Students with more previous
# failures generally have lower final grades. This makes sense because
# previous failures can indicate earlier issues.


plt.figure(figsize=(8, 5))
plt.scatter(df_clean["studytime"], df_clean["G3"], alpha=0.6)
plt.title("Weekly Study Time Category vs Final Math Grade")
plt.xlabel("Study Time Category")
plt.ylabel("Final Math Grade (G3)")
plt.savefig(os.path.join("outputs", "studytime_vs_g3.png"))
plt.close()

# I expected study time to have a stronger positive relationship with G3,
# but the relationship is fairly weak. This is surprising because studying
# more might be expected to lead directly to higher grades.
#
# Failures having a stronger relationship than study time suggests that
# previous academic performance may tell us more about final performance
# than study time by itself. Also, failure may already hold information such as study time and more.


# --- Task 4: Baseline Model ---

X = df_clean[["failures"]].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
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

# The negative slope means that for every additional previous failure,
# the model predicts the student's G3 grade will decrease by about the
# slope amount in grade points.
#
# Since G3 is only measured from 0 to 20, losing even a few grade points
# is a meaningful change.
#
# The RMSE tells us the typical prediction error in G3 grade points.
# An error around 3 points would be fairly large on a 0-20 grade scale.
#
# The R² is low, meaning failures alone do not explain very much of the
# variation in students' final grades.


# --- Task 5: Full Model ---

feature_cols = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "absences",
    "freetime",
    "goout",
    "Walc",
    "schoolsup",
    "internet",
    "higher",
    "activities",
    "sex"
]

X = df_clean[feature_cols].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
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

print("\nCoefficients:")

for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")


# Some coefficients surprised me, especially variables such as schoolsup
# and internet having larger effects than I originally expected.
# A coefficient does not automatically mean that the feature causes G3
# to change because other variables may be related to both features.
#
# The full model improves on the baseline because it uses information
# about the student's academics, behavior, background, and resources
# instead of using failures alone.
#
# The difference between train R² and test R² tells us how well the model
# generalizes. If train R² were much higher than test R², that could mean
# the model is fitting the training students better than unseen students.
#
# For a production model, I would probably keep features such as failures,
# absences, studytime, and other features that consistently help prediction.
# I would consider dropping features such as activities or traveltime if
# testing showed they add very little predictive value.
#
# I would not choose features only from coefficient size because the
# variables use different measurement scales.


# --- Task 6: Predicted vs Actual ---

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, y_test, alpha=0.6)

min_val = min(y_pred.min(), y_test.min())
max_val = max(y_pred.max(), y_test.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val]
)

plt.title("Predicted vs Actual Final Math Grades")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")

plt.savefig(os.path.join("outputs", "predicted_vs_actual.png"))
plt.close()


# Find largest positive and negative model coefficients.
coefficients = dict(zip(feature_cols, model.coef_))

largest_positive_feature = max(coefficients, key=coefficients.get)
largest_negative_feature = min(coefficients, key=coefficients.get)

largest_positive_coef = coefficients[largest_positive_feature]
largest_negative_coef = coefficients[largest_negative_feature]


print("\nSummary")
print("Filtered Dataset Size:", len(df_clean))
print("Test Set Size:", len(y_test))
print("Full Model RMSE:", rmse)
print("Full Model Test R²:", test_r2)

print(
    "Largest Positive Coefficient:",
    largest_positive_feature,
    largest_positive_coef
)

print(
    "Largest Negative Coefficient:",
    largest_negative_feature,
    largest_negative_coef
)


# The filtered dataset contains the students who actually received a final
# grade. The test set is 20% of this dataset and contains students that
# were not used to train the model.
#
# RMSE tells us approximately how many G3 grade points the predictions are
# usually away from the actual grade. Because G3 only ranges from 0 to 20,
# even an error of a few points can matter.
#
# R² tells us how much of the variation in final grades the model can
# explain. A low R² means there is still a lot about student performance
# that these features do not explain.
#
# The largest positive and negative coefficients are printed above.
# These coefficients show the largest changes in predicted G3 for a
# one-unit increase in those variables while the other model variables
# are held constant. They should not be called correlations.
#
# One surprising result to me was how little some features such as
# freetime contributed compared with academic history such as failures.
#
# In the predicted-vs-actual graph, points close to the diagonal are
# better predictions. Points above the diagonal have an actual grade
# higher than predicted, so the model underpredicted them. Points below
# the diagonal have an actual grade lower than predicted, so the model
# overpredicted them.


# --- G1 Extension ---

feature_cols_g1 = [
    "failures",
    "Medu",
    "Fedu",
    "studytime",
    "higher",
    "schoolsup",
    "internet",
    "sex",
    "freetime",
    "activities",
    "traveltime",
    "G1"
]

X = df_clean[feature_cols_g1].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

g1_model = LinearRegression()
g1_model.fit(X_train, y_train)

y_pred_g1 = g1_model.predict(X_test)

g1_r2 = r2_score(y_test, y_pred_g1)
g1_rmse = np.sqrt(mean_squared_error(y_test, y_pred_g1))

print("\nModel Including G1")
print("Test R²:", g1_r2)
print("RMSE:", g1_rmse)


# G1 is useful for prediction because a student's first-period grade can
# be a strong indicator of how they are currently performing in the class.
# It does not necessarily cause G3, though. Both G1 and G3 could reflect
# things such as academic ability, study habits, attendance, or motivation.
#
# This model could be useful for intervention after G1 becomes available
# because educators could identify students whose early grades suggest
# they may struggle by the end of the course.
#
# However, it cannot be used for truly early intervention before G1 exists.
# If educators wanted to intervene before the first-period grade, they
# would need to rely on information available earlier, such as previous
# failures, attendance history, study habits, previous grades, or other
# student background information.

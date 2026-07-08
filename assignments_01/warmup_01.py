# --- Pandas ---

# Pandas Q1
from re import sub
import pandas as pd

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

print(f"First 3 rows:{df.head(3)}")
print(f"Shape: {df.shape}")
print(f"Column types: {df.dtypes}")


# Pandas Q2

print(f"Students who passed and have a grade above 80: {df[(df['grade']>80) & (df['passed'] == True)]}")

# Pandas Q3

df['grade_curved'] = df['grade'] + 5
print(f"Curved grades: {df.head()}")

# Pandas Q4

df['name_upper'] = df['name'].str.upper()
print(f"Names in uppercase and normal name: {df[['name','name_upper']].head()}")

# Pandas Q5

print(f"Mean grade by city: {df.groupby('city')['grade'].mean()}")

# Pandas Q6

df.loc[df['city'] == 'Austin', 'city']  = 'Houston'
print(f"Confirming change: {df.head()}")

# Pandas Q7

df_sorted = df.sort_values(by='grade', ascending=False)

print(f"Sorted by grad: {df_sorted.head(3)}")


# --- Numpy ---

# Numpy Q1

import numpy as np

new_array = np.array([10, 20, 30, 40, 50])

print(f"Shape: {new_array.shape}")
print(f"Shape: {new_array.ndim}")
print(f"Array type: {new_array.dtype}")

# Numpy Q2

arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print(f"Shape: {arr.shape}")
print(f"Array type: {arr.size}")

# Numpy Q3

sliced_arr = arr[0:2,0:2]

print(f"Sliced array 2x2: {sliced_arr}")

# Numpy Q4

first_arr = np.zeros((3,4))
second_arr = np.ones((2,5))

print(f"First arr: {first_arr}")
print(f"Second arr: {second_arr}")

# Numpy Q5

arranged_arr = np.arange(0, 50, 5)

print(f"The array: {arranged_arr}")
print(f"Shape: {arranged_arr.shape}")
print(f"Mean: {arranged_arr.mean()}")
print(f"Sum: {arranged_arr.sum()}")
print(f"STD: {arranged_arr.std()}")

# Numpy Q6

dist = np.random.normal(loc=0,scale=1,size=200)

print(f"Mean: {dist.mean()}")
print(f"STD: {dist.std()}")

# --- Matplotlib  ---

# Matplotlib  Q1

import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.plot(x,y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Matplotlib  Q2

subjects = ["Math", "Science", "English", "History"]
scores  = [88, 92, 75, 83]


plt.bar(subjects,scores)
plt.title("Subject Scores")
plt.xlabel("subjects")
plt.ylabel("scores")
plt.show()

# Matplotlib  Q3

x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1,y1, color="green")
plt.scatter(x2,y2, color="blue")
plt.title("Scatter",)
plt.xlabel("x")
plt.ylabel("y")
plt.legend(labels = ['First','Second'])
plt.show()

# Matplotlib  Q4

fig, (xaxes,subjectsaxes) = plt.subplots(1,2)
xaxes.plot(x,y)
xaxes.set_title("Line plot")
subjectsaxes.bar(subjects,scores)
subjectsaxes.set_title("Bar plot")
plt.tight_layout()
plt.show()

# --- Descriptive Stats ---

# Descriptive Stats Q1

data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
datanp = np.array(data)

print(f"Mean: {datanp.mean()}")
print(f"Median: {np.median(datanp)}")
print(f"STD: {datanp.std()}")
print(f"Variance: {datanp.var()}")

# Descriptive Stats Q2

dist2 = np.random.normal(65, 10, 500)
plt.hist(dist2, bins=20)
plt.title("Distribution of Scores")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.boxplot([group_a,group_b],label=["Group A", "Group B"])
plt.title("Score Comparison")
plt.show()

# Descriptive Stats Q4

normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

plt.boxplot([normal_data,skewed_data],label=["Normal Data", "Skewed Data"])
plt.title("Distribution Comparison")
plt.show()

#The most skewed distribution is the second because the median is right skewed
#This makes sense because the exponential would rapidly increase the magnitude of the larger values
#The best measure of central tendancy for the first dist is average there isn't much influence from outliers
#The best measure of central tendancy for the second dist is median even with influence from outliers it will stay the least influenced


# Descriptive Stats Q5

data1 = [10, 12, 12, 16, 18]

data2 = [10, 12, 12, 16, 150]

data1np = np.array(data1)

data2np = np.array(data2)



print(f"Mean: {data1np.mean()}")
print(f"Median: {np.median(data1np)}")
values1, counts1 = np.unique(data1np, return_counts=True)
mode1 = values1[np.argmax(counts1)]
print(f"Mode: {mode1}")

print(f"Mean: {data2np.mean()}")
print(f"Median: {np.median(data2np)}")
values2, counts2 = np.unique(data2np, return_counts=True)
mode2 = values2[np.argmax(counts2)]
print(f"Mode: {mode2}")

#Due to the outlier of 150 in data 2 the mean is so much higher than data 1

# --- Hypothesis Testing ---

# Hypothesis Testing Q1

from scipy import stats

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

group_anp = np.array(group_a)
group_bnp = np.array(group_b)

tTest = stats.ttest_ind(group_anp,group_bnp)

print(f"T-statistic: {tTest.statistic}")
print(f"P-value: {tTest.pvalue}")

# Hypothesis Testing Q2

if(tTest.pvalue<0.05):
    print("This test is statistically significant")
else:
    print("This test is not statistically significant")

# Hypothesis Testing Q3

before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

tTestPaired = stats.ttest_rel(before, after)

print(f"T-statistic: {tTestPaired.statistic}")
print(f"P-value: {tTestPaired.pvalue}")

# Hypothesis Testing Q4

scores = [72, 68, 75, 70, 69, 74, 71, 73]
scoresnp = np.array(scores)
tTest1Samp = stats.ttest_1samp(scoresnp, popmean=70)

print(f"T-statistic: {tTest1Samp.statistic}")
print(f"P-value: {tTest1Samp.pvalue}")

# Hypothesis Testing Q5

tTestCompare = stats.ttest_ind(group_a,group_b,alternative="less")

print(f"P-value: {tTestCompare.pvalue}")

# Hypothesis Testing Q6
print(
    "The T stat is negative therefore group A's mean is less than group B's"
    "The difference in these means are statstically significant therefore"
    "the means do have a large difference and the null hypothesis being no difference is null."
    )

# --- Correlation ---

# Correlation Q1

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

matrixC = np.corrcoef(x,y)
corrc = matrixC[0,1]

print(f"Full Correlation Matrix: {matrixC}")
print(f"Correlation Coefficient: {corrc}")

# I would expect a positive correltion because both are increasing in the same direction
# and at the same rate so they will be linearly pretty close


# Correlation Q2

from scipy.stats import pearsonr

x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

rCorr = pearsonr(x,y)
print(f"T-statistic: {rCorr.statistic}")
print(f"P-value: {rCorr.pvalue}")

# Correlation Q3

people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)

frameCorr = df.corr()

print(f"Correlation Matrix: {frameCorr}")

# Correlation Q4

x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.scatter(x,y)
plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Correlation Q5

import seaborn as sns

sns.heatmap(frameCorr, annot = True)
plt.title("Correlation Heatmap")
plt.show()

# --- Pipeline ---

# Pipeline Q1

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])


def create_series(arr):
    return pd.Series(arr, name = 'values')

def clean_data(series):
    return series.dropna(ignore_index=True)

def summarize_data(series):
    return {"mean":series.agg("mean"),"median":series.agg("median"),"std":series.agg("std"),"mode":series.mode()[0]}

def data_pipeline(arr):
    new_series = create_series(arr)
    cleaned_series = clean_data(new_series)
    summary = summarize_data(cleaned_series)
    for key in summary:
        print(f"Key {key} and value {summary[key]}")

data_pipeline(arr)
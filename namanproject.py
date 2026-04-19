# ============================================================
# PYTHON PROJECT: India City-wise Air Pollution Data
# Source: Real-time Pollution Monitoring Stations, India
# ============================================================

# ===========================
# 1. IMPORT LIBRARIES
# ===========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind
# ===========================
# 2. LOAD DATA
# ===========================
df = pd.read_csv("project.csv")
# ===========================
# 3. EXPLORATORY DATA ANALYSIS
# ===========================
print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== LAST 5 ROWS =====")
print(df.tail())

print("\n===== SHAPE OF DATA =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns)

print("\n===== DATA INFO =====")
df.info()

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# ===========================
# 4. DATA CLEANING
# ===========================

df = df.dropna(subset=['pollutant_min', 'pollutant_max', 'pollutant_avg'])

print("\n===== SHAPE AFTER CLEANING =====")
print(df.shape)

# ===========================
# 5. OUTLIER VISUALIZATION
# ===========================

# histogram
plt.figure(figsize=(8, 5))
sns.histplot(df['pollutant_avg'], kde=True)
plt.title("Histogram of Pollutant Avg\nObjective: Understand distribution of average pollution levels")
plt.xlabel("Pollutant Avg")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.show()

# boxplot
plt.figure(figsize=(6, 8))
sns.boxplot(y=df['pollutant_avg'], color='steelblue')
plt.title("Boxplot of Pollutant Avg\nObjective: Detect outliers in average pollution levels")
plt.ylabel("Pollutant Avg")
plt.grid(alpha=0.3)
plt.show()

# ===========================
# 6. OUTLIER REMOVAL (IQR METHOD)
# ===========================

Q1 = df['pollutant_avg'].quantile(0.25)
Q3 = df['pollutant_avg'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print("\nLower Bound:", lower)
print("Upper Bound:", upper)

df_clean = df[(df['pollutant_avg'] >= lower) & (df['pollutant_avg'] <= upper)]

print("\n===== SHAPE AFTER OUTLIER REMOVAL =====")
print(df_clean.shape)


# ============================================================
# 7. PAIR PLOT(TO SHOW OUTLIERS AND RELATIONSHIPS)
# OBJECTIVE:-To visualize relationships between multiple numerical variables
# (pollutant_min, pollutant_max, pollutant_avg).
# This helps in identifying correlations, patterns, distributions,
# and detecting outliers in pollution data.
# ============================================================


sns.pairplot(df_clean[['pollutant_min', 'pollutant_max', 'pollutant_avg']])
plt.suptitle("Pair Plot: Pollution Variables\nObjective: Visualize relationships between numerical variables", y=1.02)
plt.show()

# ============================================================
# OBJECTIVE 1 : COUNT PLOT
# To analyze how many records exist for each pollutant type.
# This helps to understand which pollutants are most monitored
# across India.
# ============================================================

plt.figure(figsize=(10, 5))
sns.countplot(x='pollutant_id', data=df_clean, order=df_clean['pollutant_id'].value_counts().index,palette='viridis')
plt.title("Count Plot: Records by Pollutant Type\nObjective 1: Analyze frequency of each pollutant monitored")
plt.xlabel("Pollutant Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ============================================================
# OBJECTIVE 2 : LINEAR REGRESSION GRAPH (SCATTER + REGRESSION LINE)
# To analyze and model the relationship between pollutant_min
# and pollutant_max using a regression line, helping to
# understand trends and predict maximum pollution levels.
# ============================================================

X = df_clean[['pollutant_min']]
y = df_clean['pollutant_max']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.figure(figsize=(8,5))
plt.scatter(X, y, alpha=0.5, label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.title("Linear Regression: Pollutant Min vs Max\nObjective 2: Analyze trend and predict maximum pollution")
plt.xlabel("Pollutant Min")
plt.ylabel("Pollutant Max")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
# ============================================================
# OBJECTIVE 3 : BAR GRAPH
# To identify the top 10 states with highest average pollution.
# This helps to find which states in India are most polluted
# and need immediate attention.
# ============================================================

top_states = df_clean.groupby('state')['pollutant_avg'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_states.index, y=top_states.values)
plt.title("Bar Graph: Top 10 Most Polluted States\nObjective 3: Identify states with highest average pollution levels")
plt.xlabel("State")
plt.ylabel("Average Pollution Level")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

# ============================================================
# OBJECTIVE 4 : CORRELATION HEATMAP
# To analyze the correlation between numerical variables.
# This helps to understand how minimum, maximum and average
# pollution levels are related to each other.
# ============================================================

num_df = df_clean[['pollutant_min', 'pollutant_max', 'pollutant_avg']]
corr = num_df.corr()

print("\n===== CORRELATION MATRIX =====")
print(corr)

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Heatmap: Correlation Matrix\nObjective 4: Analyze relationship between pollution variables")
plt.tight_layout()
plt.show()

# ============================================================
# OBJECTIVE 5 : PIE CHART
# To analyze the proportion of records for each pollutant type.
# This helps to compare which pollutants are most frequently
# measured across all monitoring stations in India.
# ============================================================

pollutant_counts = df_clean['pollutant_id'].value_counts()

plt.figure(figsize=(7, 7))
plt.pie(pollutant_counts.values, labels=pollutant_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Pie Chart: Distribution of Pollutant Types\nObjective 5: Compare proportion of each pollutant monitored")
plt.show()

# ===========================
# 8. HYPOTHESIS TESTING
# ===========================
# Objective:
# To test whether there is a significant difference in average
# pollution levels between PM2.5 and PM10.
# H0: There is no significant difference in avg pollution between PM2.5 and PM10.
# H1: There is a significant difference in avg pollution between PM2.5 and PM10.

pm25 = df_clean[df_clean['pollutant_id'] == 'PM2.5']['pollutant_avg']
pm10 = df_clean[df_clean['pollutant_id'] == 'PM10']['pollutant_avg']

t_stat, p_value = ttest_ind(pm25, pm10)

print("\n===== HYPOTHESIS TESTING RESULT =====")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value    : {p_value:.4f}")

if p_value < 0.05:
    print("Conclusion: Reject H0 → There IS a significant difference in pollution levels between PM2.5 and PM10.")
else:
    print("Conclusion: Fail to Reject H0 → There is NO significant difference.")
    plt.tight_layout()
    plt.show()

    print("\n✅ VALIDATION COMPLETE — Model is Justified!")

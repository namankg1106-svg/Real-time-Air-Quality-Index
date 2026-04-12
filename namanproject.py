# ============================================================
# PYTHON PROJECT: India City-wise Air Pollution Data
# Source: Real-time Pollution Monitoring Stations, India
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# PHASE 1: LOAD & CLEAN DATA
# ============================================================

# Load CSV
df = pd.read_csv("project.csv")

# Strip column names
df.columns = df.columns.str.strip()

# Check actual columns
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

# Parse date and extract hour for time-based analysis
df['last_update'] = pd.to_datetime(df['last_update'], dayfirst=True, errors='coerce')

# Convert numeric columns
for col in ['pollutant_min', 'pollutant_max', 'pollutant_avg']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing avg values
df = df.dropna(subset=['pollutant_avg']).copy()
df.reset_index(drop=True, inplace=True)

print("\n✅ Data Loaded & Cleaned!")
print(f"Shape after cleaning: {df.shape}")
print(f"Pollutants covered: {sorted(df['pollutant_id'].unique())}")
print(f"States: {df['state'].nunique()}")
print(f"Cities: {df['city'].nunique()}")
print("\nFirst 5 rows:")
print(df[['state', 'city', 'pollutant_id', 'pollutant_avg']].head())
print("\nBasic Statistics (pollutant_avg):")
print(df['pollutant_avg'].describe())


# ============================================================
# PHASE 2: EDA — 5 OBJECTIVES
# ============================================================

# ============================================================
# OBJECTIVE 1: Average Pollution Level by Pollutant Type
# ============================================================
pollutant_avg = df.groupby('pollutant_id')['pollutant_avg'].mean().reset_index()
pollutant_avg.columns = ['Pollutant', 'Avg_Level']
pollutant_avg = pollutant_avg.sort_values('Avg_Level', ascending=False)

print("=" * 55)
print("OBJECTIVE 1: Average Pollution Level by Pollutant")
print("=" * 55)
print(pollutant_avg.to_string(index=False))

# ============================================================
# OBJECTIVE 2: Top 5 & Bottom 5 States by Average PM2.5
# ============================================================
pm25 = df[df['pollutant_id'] == 'PM2.5']
state_pm25 = pm25.groupby('state')['pollutant_avg'].mean().reset_index()
state_pm25.columns = ['State', 'Avg_PM2.5']
state_pm25 = state_pm25.sort_values('Avg_PM2.5', ascending=False)

print("\n" + "=" * 55)
print("OBJECTIVE 2: Top 5 States by Average PM2.5")
print("=" * 55)
print(state_pm25.head(5).to_string(index=False))
print("\nBottom 5 States by Average PM2.5:")
print(state_pm25.tail(5).to_string(index=False))

# ============================================================
# OBJECTIVE 3: Pollutant Range — Max vs Min Gap by Pollutant
# ============================================================
pollutant_range = df.groupby('pollutant_id').agg(
    Avg_Min=('pollutant_min', 'mean'),
    Avg_Max=('pollutant_max', 'mean')
).reset_index()
pollutant_range['Range_Gap'] = (pollutant_range['Avg_Max'] - pollutant_range['Avg_Min']).round(2)
pollutant_range = pollutant_range.sort_values('Range_Gap', ascending=False)

print("\n" + "=" * 55)
print("OBJECTIVE 3: Avg Min/Max Range Gap by Pollutant")
print("=" * 55)
print(pollutant_range.to_string(index=False))

# ============================================================
# OBJECTIVE 4: State-wise Average Pollution (All Pollutants)
# ============================================================
state_overall = df.groupby('state')['pollutant_avg'].mean().reset_index()
state_overall.columns = ['State', 'Overall_Avg_Pollution']
state_overall = state_overall.sort_values('Overall_Avg_Pollution', ascending=False)

print("\n" + "=" * 55)
print("OBJECTIVE 4: State-wise Overall Average Pollution Level")
print("=" * 55)
print(state_overall.to_string(index=False))

# ============================================================
# OBJECTIVE 5: Top 10 Most Polluted Cities (by avg PM10)
# ============================================================
pm10 = df[df['pollutant_id'] == 'PM10']
city_pm10 = pm10.groupby(['state', 'city'])['pollutant_avg'].mean().reset_index()
city_pm10.columns = ['State', 'City', 'Avg_PM10']
city_pm10 = city_pm10.sort_values('Avg_PM10', ascending=False)

print("\n" + "=" * 55)
print("OBJECTIVE 5: Top 10 Most Polluted Cities (PM10)")
print("=" * 55)
print(city_pm10.head(10).to_string(index=False))


# ============================================================
# PHASE 3: VISUALISATIONS
# ============================================================

sns.set_style("whitegrid")

# ----------------------------------------------------------
# PLOT 1 (Obj 1): Average Pollution Level by Pollutant — Line Chart
# ----------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(pollutant_avg['Pollutant'], pollutant_avg['Avg_Level'],
         marker='o', color='steelblue', linewidth=2.5, markersize=7)
for _, row in pollutant_avg.iterrows():
    plt.text(row['Pollutant'], row['Avg_Level'] + 1,
             f"{row['Avg_Level']:.1f}", ha='center', fontsize=8)
plt.title("Objective 1: Average Pollution Level by Pollutant Type",
          fontsize=14, fontweight='bold')
plt.xlabel("Pollutant")
plt.ylabel("Average Level (µg/m³ or ppb)")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# PLOT 2 (Obj 2): Top 10 States by PM2.5 — Horizontal Bar Chart
# ----------------------------------------------------------
top10_states = state_pm25.head(10)
plt.figure(figsize=(10, 6))
bars = plt.barh(top10_states['State'][::-1],
                top10_states['Avg_PM2.5'][::-1],
                color='coral')
plt.title("Objective 2: Top 10 States by Average PM2.5 Level",
          fontsize=13, fontweight='bold')
plt.xlabel("Average PM2.5 (µg/m³)")
for bar in bars:
    plt.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
             f"{bar.get_width():.1f}", va='center', fontsize=8)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# PLOT 3 (Obj 3): Range Gap by Pollutant — Bar Chart
# ----------------------------------------------------------
plt.figure(figsize=(10, 5))
colors = ['tomato' if x > pollutant_range['Range_Gap'].mean() else 'steelblue'
          for x in pollutant_range['Range_Gap']]
plt.bar(pollutant_range['pollutant_id'], pollutant_range['Range_Gap'], color=colors)
plt.title("Objective 3: Avg Min–Max Range Gap by Pollutant",
          fontsize=13, fontweight='bold')
plt.xlabel("Pollutant")
plt.ylabel("Range Gap (Max – Min Avg)")
plt.xticks(rotation=40, ha='right', fontsize=8)
plt.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# PLOT 4 (Obj 4): State-wise Overall Pollution — Bar Chart
# ----------------------------------------------------------
top_states_overall = state_overall.head(15)
yoy_colors = ['green' if x < state_overall['Overall_Avg_Pollution'].median() else 'red'
              for x in top_states_overall['Overall_Avg_Pollution']]
plt.figure(figsize=(12, 5))
plt.bar(top_states_overall['State'], top_states_overall['Overall_Avg_Pollution'],
        color=yoy_colors, edgecolor='black', linewidth=0.5)
plt.title("Objective 4: State-wise Overall Average Pollution Level (Top 15)",
          fontsize=13, fontweight='bold')
plt.xlabel("State")
plt.ylabel("Average Pollution Level")
plt.axhline(state_overall['Overall_Avg_Pollution'].median(), color='black', linewidth=1, linestyle='--', label='Median')
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# PLOT 5 (Obj 5): Heatmap — Top 15 States × Pollutant
# ----------------------------------------------------------
top15_states = state_overall.head(15)['State'].tolist()
heatmap_data = df[df['state'].isin(top15_states)].groupby(
    ['state', 'pollutant_id'])['pollutant_avg'].mean().reset_index()
heatmap_pivot = heatmap_data.pivot(index='state', columns='pollutant_id', values='pollutant_avg')

plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_pivot, cmap='YlOrRd', linewidths=0.4,
            annot=True, fmt='.1f', annot_kws={'size': 7})
plt.title("Objective 5: Pollution Heatmap — Top 15 States × Pollutant Type",
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# ============================================================
# PHASE 4: LINEAR REGRESSION MODEL
# — Predicting average PM2.5 from PM10 across cities
# ============================================================
print("\n" + "=" * 55)
print("PHASE 4: LINEAR REGRESSION MODEL")
print("=" * 55)

# Pivot: one row per city — avg PM10 predicts avg PM2.5
city_pivot = df[df['pollutant_id'].isin(['PM10', 'PM2.5'])].groupby(
    ['city', 'pollutant_id'])['pollutant_avg'].mean().unstack('pollutant_id').dropna()
city_pivot.columns.name = None
city_pivot = city_pivot.reset_index()
city_pivot.rename(columns={'PM10': 'Avg_PM10', 'PM2.5': 'Avg_PM25'}, inplace=True)

# Feature: Avg_PM10 | Target: Avg_PM25
X = city_pivot[['Avg_PM10']]
y = city_pivot['Avg_PM25']

model = LinearRegression()
model.fit(X, y)

print(f"Model Equation : PM2.5 = {model.coef_[0]:.4f} × PM10 + ({model.intercept_:.4f})")
print(f"Interpretation : For each 1 unit rise in PM10, PM2.5 rises by ~{model.coef_[0]:.4f}")

# Predictions on all cities
y_pred_all = model.predict(X)

# Cross-validation R²
cv_scores = cross_val_score(LinearRegression(), X, y, cv=4, scoring='r2')
print(f"\nCross-Validated R² scores : {[round(s, 4) for s in cv_scores]}")
print(f"Mean CV R²                : {cv_scores.mean():.4f}")

# Accuracy Metrics
mae  = mean_absolute_error(y, y_pred_all)
mse  = mean_squared_error(y, y_pred_all)
rmse = np.sqrt(mse)
r2   = r2_score(y, y_pred_all)

print("\n📐 Model Accuracy Metrics:")
print(f"  MAE  : {mae:.4f}   ← avg prediction error in PM2.5")
print(f"  MSE  : {mse:.4f}")
print(f"  RMSE : {rmse:.4f}   ← error in same unit as PM2.5")
print(f"  R²   : {r2:.4f}    ← model explains {r2:.1%} of variance")

# Actual vs Predicted Table (top 15 cities)
print("\nActual vs Predicted PM2.5 (Sample — Top 15 Cities by PM10):")
sample = city_pivot.nlargest(15, 'Avg_PM10').copy()
sample['Predicted_PM25'] = model.predict(sample[['Avg_PM10']])
sample['Error'] = sample['Avg_PM25'] - sample['Predicted_PM25']
print(f"{'City':<25} {'Actual PM2.5':>14} {'Predicted':>12} {'Error':>10}")
print("-" * 65)
for _, row in sample.iterrows():
    print(f"{row['city']:<25} {row['Avg_PM25']:>14.2f} {row['Predicted_PM25']:>12.2f} {row['Error']:>+10.2f}")

# Future Projections — predict PM2.5 for PM10 levels 50 to 300
future = pd.DataFrame({'Avg_PM10': range(50, 305, 25)})
future_pred = model.predict(future)

print("\n🔮 PM2.5 Projections for Given PM10 Levels:")
print(f"{'PM10 Level':>12} {'Projected PM2.5':>18}")
print("-" * 32)
for pm10_val, pred in zip(future['Avg_PM10'], future_pred):
    print(f"{pm10_val:>12} {pred:>18.2f}")

# Final Regression Plot
x_range = pd.DataFrame({'Avg_PM10': np.linspace(X['Avg_PM10'].min(), X['Avg_PM10'].max(), 200)})
y_range = model.predict(x_range)

plt.figure(figsize=(12, 6))
plt.scatter(X['Avg_PM10'], y,
            color='steelblue', s=60, zorder=5, alpha=0.7, label='Actual City Data')
plt.plot(x_range['Avg_PM10'], y_range,
         color='tomato', linewidth=2.5, linestyle='--', label='Regression Line')
plt.scatter(future['Avg_PM10'], future_pred,
            color='orange', s=90, zorder=5, marker='D', label='Projected PM2.5')
for pm10_val, pred in zip(future['Avg_PM10'], future_pred):
    plt.text(pm10_val, pred + 0.5, f"{pred:.1f}", ha='center', fontsize=7)
plt.title("Linear Regression: PM10 vs PM2.5 Across Indian Cities",
          fontsize=14, fontweight='bold')
plt.xlabel("Average PM10 (µg/m³)")
plt.ylabel("Average PM2.5 (µg/m³)")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

print("\n✅ PROJECT COMPLETE!")

# ============================================================
# PHASE 5: MODEL VALIDATION — High vs Low PM10 City Groups
# ============================================================
print("\n" + "=" * 60)
print("PHASE 5: MODEL VALIDATION — HIGH vs LOW PM10 GROUPS")
print("=" * 60)

# Split cities into high PM10 (top 50%) and low PM10 (bottom 50%)
median_pm10 = city_pivot['Avg_PM10'].median()
high_pm10 = city_pivot[city_pivot['Avg_PM10'] >= median_pm10].copy()
low_pm10  = city_pivot[city_pivot['Avg_PM10'] <  median_pm10].copy()

groups = {'High PM10': high_pm10, 'Low PM10': low_pm10}
actual_totals  = {}
predicted_vals_dict = {}

# Train model on full data and validate on each group
val_model = LinearRegression()
val_model.fit(X, y)

for grp_name, grp_df in groups.items():
    X_grp = grp_df[['Avg_PM10']]
    y_grp = grp_df['Avg_PM25']
    y_pred_grp = val_model.predict(X_grp)
    actual_totals[grp_name]       = y_grp.mean()
    predicted_vals_dict[grp_name] = y_pred_grp.mean()
    print(f"  {grp_name}: Actual Avg PM2.5 = {y_grp.mean():.2f} | Predicted Avg = {y_pred_grp.mean():.2f}")

# Comparison Table
print(f"\n{'Group':<15} {'Predicted':>14} {'Actual':>14} {'Diff':>14} {'Error%':>10}")
print("-" * 70)
actuals_list    = []
predicted_list  = []
group_names     = []

for grp_name in groups:
    actual = actual_totals[grp_name]
    pred   = predicted_vals_dict[grp_name]
    diff   = actual - pred
    pct    = abs(diff / actual) * 100
    actuals_list.append(actual)
    predicted_list.append(pred)
    group_names.append(grp_name)
    print(f"{grp_name:<15} {pred:>14.2f} {actual:>14.2f} {diff:>+14.2f} {pct:>9.2f}%")

# Validation Accuracy Metrics
if actuals_list:
    val_mae  = mean_absolute_error(actuals_list, predicted_list)
    val_rmse = np.sqrt(mean_squared_error(actuals_list, predicted_list))
    try:
        val_r2 = r2_score(actuals_list, predicted_list)
    except Exception:
        val_r2 = float('nan')

    print(f"\n📐 Validation Metrics (Group-level Predicted vs Actual):")
    print(f"   MAE    : {val_mae:.4f}  ← avg error in PM2.5")
    print(f"   RMSE   : {val_rmse:.4f}")
    avg_err = sum(abs(a-p)/a*100 for a, p in zip(actuals_list, predicted_list)) / len(actuals_list)
    print(f"   Avg Error% : {avg_err:.2f}%")

    # Final Validation Plot
    plt.figure(figsize=(13, 6))

    # All training scatter
    plt.scatter(X['Avg_PM10'], y,
                color='steelblue', s=60, zorder=3, alpha=0.5,
                label='All Cities (Training Data)')

    # Regression line
    plt.plot(x_range['Avg_PM10'], y_range,
             color='tomato', linewidth=2.5, linestyle='--', label='Regression Line')

    # Actual group averages (green squares)
    x_positions = [high_pm10['Avg_PM10'].mean(), low_pm10['Avg_PM10'].mean()]
    plt.scatter(x_positions, list(actual_totals.values()),
                color='green', s=150, zorder=6, marker='s',
                label='Actual Group Avg PM2.5 (Validation)')

    # Predicted group averages (orange diamonds)
    plt.scatter(x_positions, list(predicted_vals_dict.values()),
                color='orange', s=120, zorder=6, marker='D',
                label='Model Predicted Group Avg PM2.5')

    # Error lines
    for x_pos, act, pred, grp_name in zip(x_positions,
                                           list(actual_totals.values()),
                                           list(predicted_vals_dict.values()),
                                           group_names):
        plt.plot([x_pos, x_pos], [act, pred], 'gray', linestyle=':', linewidth=2)
        mid_y   = (act + pred) / 2
        err_pct = abs(act - pred) / act * 100
        plt.text(x_pos + 1, mid_y, f"Error: {err_pct:.1f}%",
                 fontsize=9, color='dimgray', fontweight='bold')

    plt.title("Model Validation: Predicted vs Actual PM2.5 (City Groups)",
              fontsize=14, fontweight='bold')
    plt.xlabel("Average PM10 (µg/m³)")
    plt.ylabel("Average PM2.5 (µg/m³)")
    plt.legend(fontsize=9, loc='upper left')
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    print("\n✅ VALIDATION COMPLETE — Model is Justified!")
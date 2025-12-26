import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
climateData = pd.read_csv("ClimateDataset.csv")

# Extract clean data for all relevant variables
mask = (
    ~climateData['Year'].isna() & 
    ~climateData['CO2 emissions'].isna() &
    ~climateData['RainFall Mean ()'].isna() &
    ~climateData['Population, total Value'].isna()
)

years = climateData['Year'].values[mask]                            # type: ignore
co2 = climateData['CO2 emissions'].values[mask]                     # type: ignore
rainfall = climateData['RainFall Mean ()'].values[mask]             # type: ignore
population = climateData['Population, total Value'].values[mask]    # type: ignore

# 1. SIMPLE MODEL: CO2 ~ Year
X_simple = np.column_stack([np.ones_like(years), years])
beta_simple = np.linalg.inv(X_simple.T @ X_simple) @ X_simple.T @ co2
intercept_simple, slope_simple = beta_simple
error_simple = np.sum((co2 - X_simple @ beta_simple)**2)

# 2. MULTIPLE MODEL: CO2 ~ Year + Rainfall + Population
X_multi = np.column_stack([np.ones_like(years), years, rainfall, population])
beta_multi = np.linalg.inv(X_multi.T @ X_multi) @ X_multi.T @ co2
error_multi = np.sum((co2 - X_multi @ beta_multi)**2)

# Results
print("="*60)
print("LEAST SQUARES RESULTS: CO2 EMISSIONS MODELS")
print("="*60)
print("\n1. SIMPLE MODEL (CO2 ~ Year):")
print(f"   Intercept: {intercept_simple:,.2f}")
print(f"   Slope: {slope_simple:.3f} kilo tons/year")
print(f"   Least Squares Error: {error_simple:,.0f}")

print("\n2. MULTIPLE MODEL (CO2 ~ Year + Rainfall + Population):")
print(f"   Intercept: {beta_multi[0]:,.2f}")
print(f"   Year coefficient: {beta_multi[1]:.3f}")
print(f"   Rainfall coefficient: {beta_multi[2]:.3f}")
print(f"   Population coefficient: {beta_multi[3]:.6f}")
print(f"   Least Squares Error: {error_multi:,.0f}")

print("\n3. IMPROVEMENT:")
print(f"   Error reduction: {error_simple - error_multi:,.0f}")
print(f"   % Improvement: {(1 - error_multi/error_simple)*100:.1f}%")

# Visualization
plt.figure(figsize=(12, 4))

# Plot 1: Simple model
plt.subplot(1, 3, 1)
plt.scatter(years, co2, alpha=0.5, s=20)
x_line = np.array([years.min(), years.max()])
y_line = intercept_simple + slope_simple * x_line
plt.plot(x_line, y_line, 'r-', linewidth=2)
plt.xlabel('Year'); plt.ylabel('CO2 Emissions')
plt.title(f'Simple: CO2 ~ Year\nError: {error_simple:,.0f}')

# Plot 2: Multiple model predictions
plt.subplot(1, 3, 2)
co2_pred_multi = X_multi @ beta_multi
plt.scatter(co2, co2_pred_multi, alpha=0.5, s=20)
plt.plot([co2.min(), co2.max()], [co2.min(), co2.max()], 'r--')
plt.xlabel('Actual CO2'); plt.ylabel('Predicted CO2')
plt.title(f'Multiple Model Predictions\nError: {error_multi:,.0f}')


# Plot 3: Error comparison
plt.subplot(1, 3, 3)
models = ['Simple\n(Year only)', 'Multiple\n(Year+Rain+Pop)']
errors = [error_simple, error_multi]
bars = plt.bar(models, errors, color=['red', 'blue'], alpha=0.7)
plt.ylabel('Least Squares Error')
plt.title('Model Comparison')
for bar, error in zip(bars, errors):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10000,
             f'{error:,.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('saved_figures/least_squares_comparison.png', dpi=150)
plt.show()
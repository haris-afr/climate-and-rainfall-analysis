import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
climateData = pd.read_csv("ClimateDataset.csv")

# Extract clean data
mask = (
    climateData['Year'].notna() & 
    climateData['CO2 emissions'].notna() &
    climateData['Temperature'].notna() &
    climateData['Rainfall'].notna() &
    climateData['Population'].notna()
)

years = climateData['Year'].values[mask] # type: ignore
co2 = climateData['CO2 emissions'].values[mask] # type: ignore
temp = climateData['Temperature'].values[mask] # type: ignore
rain = climateData['Rainfall'].values[mask] # type: ignore
pop = climateData['Population'].values[mask] # type: ignore

print(f"Data: {len(years)} years, {years.min()} to {years.max()}")

# 1. CO2 ~ Year
X1 = np.column_stack([np.ones_like(years), years])
beta1 = np.linalg.inv(X1.T @ X1) @ X1.T @ co2
lse1 = np.sum((co2 - X1 @ beta1)**2)

# 2. CO2 ~ Year + Temperature
X2 = np.column_stack([np.ones_like(years), years, temp])
beta2 = np.linalg.inv(X2.T @ X2) @ X2.T @ co2
lse2 = np.sum((co2 - X2 @ beta2)**2)

# 3. CO2 ~ Year + Temperature + Rainfall
X3 = np.column_stack([np.ones_like(years), years, temp, rain])
beta3 = np.linalg.inv(X3.T @ X3) @ X3.T @ co2
lse3 = np.sum((co2 - X3 @ beta3)**2)

# 4. CO2 ~ Year + Temperature + Rainfall + Population
X4 = np.column_stack([np.ones_like(years), years, temp, rain, pop])
beta4 = np.linalg.inv(X4.T @ X4) @ X4.T @ co2
lse4 = np.sum((co2 - X4 @ beta4)**2)

# Results
print("\n" + "="*60)
print("LEAST SQUARES RESULTS")
print("="*60)

print(f"\n1. CO2 ~ Year:")
print(f"   Intercept: {beta1[0]:.2f}")
print(f"   Slope: {beta1[1]:.4f} M tons/year")
print(f"   LSE: {lse1:,.0f}")

print(f"\n2. CO2 ~ Year + Temperature:")
print(f"   Year: {beta2[1]:.4f}")
print(f"   Temp: {beta2[2]:.4f} M tons/°C")
print(f"   LSE: {lse2:,.0f}")
print(f"   Improvement: {lse1 - lse2:,.0f} less error")

print(f"\n3. CO2 ~ Year + Temperature + Rainfall:")
print(f"   Year: {beta3[1]:.4f}")
print(f"   Temp: {beta3[2]:.4f}")
print(f"   Rain: {beta3[3]:.4f} M tons/mm")
print(f"   LSE: {lse3:,.0f}")
print(f"   Improvement: {lse2 - lse3:,.0f} less error")

print(f"\n4. CO2 ~ Year + Temperature + Rainfall + Population:")
print(f"   Year: {beta4[1]:.4f}")
print(f"   Temp: {beta4[2]:.4f}")
print(f"   Rain: {beta4[3]:.4f}")
print(f"   Population: {beta4[4]:.8f} M tons/person")
print(f"   LSE: {lse4:,.0f}")
print(f"   Improvement: {lse3 - lse4:,.0f} less error")

print(f"\n" + "="*60)
print("MODEL COMPARISON (Lower LSE = Better)")
print("="*60)
print(f"Simple model LSE: {lse1:,.0f}")
print(f"With temperature: {lse2:,.0f} ({((lse1-lse2)/lse1*100):.1f}% less error)")
print(f"With temp+rain: {lse3:,.0f} ({((lse1-lse3)/lse1*100):.1f}% less error)")
print(f"With all factors: {lse4:,.0f} ({((lse1-lse4)/lse1*100):.1f}% less error)")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: LSE comparison
ax = axes[0]
models = ['Year', '+Temp', '+Temp+Rain', 'All']
lse_values = [lse1, lse2, lse3, lse4]
bars = ax.bar(models, lse_values, color=['blue', 'green', 'orange', 'red'])
ax.set_ylabel('Least Squares Error')
ax.set_title('Model Comparison\n(Lower LSE = Better Fit)')
for bar, lse_val in zip(bars, lse_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
            f'{lse_val:,.0f}', ha='center', va='bottom', fontsize=9)

# Plot 2: Model predictions
ax = axes[1]
co2_pred_full = X4 @ beta4
ax.scatter(years, co2, alpha=0.3, s=20, label='Actual')
ax.scatter(years, co2_pred_full, alpha=0.5, s=15, label='Predicted', color='red')
ax.set_xlabel('Year')
ax.set_ylabel('CO2 Emissions (M tons)')
ax.set_title(f'Best Model Predictions\nLSE: {lse4:,.0f}')
ax.legend()

# Plot 3: Coefficients
ax = axes[2]
coeffs = beta4[1:]  # Exclude intercept
names = ['Year', 'Temp', 'Rain', 'Pop']
colors = ['blue', 'red', 'cyan', 'purple']
bars = ax.bar(names, coeffs, color=colors, alpha=0.7)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_ylabel('Coefficient Value')
ax.set_title('Full Model Coefficients')
for bar, coeff in zip(bars, coeffs):
    ax.text(bar.get_x() + bar.get_width()/2, 
            coeff + (0.1 if coeff >= 0 else -0.1),
            f'{coeff:.3f}' if abs(coeff) > 0.001 else f'{coeff:.6f}',
            ha='center', va='bottom' if coeff >= 0 else 'top')

plt.tight_layout()
plt.savefig('saved_figures/lse_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Error breakdown
print("\n" + "="*60)
print("ERROR ANALYSIS")
print("="*60)

# Calculate average error per data point
avg_error_simple = np.sqrt(lse1/len(years))
avg_error_full = np.sqrt(lse4/len(years))

print(f"\nAverage error per year:")
print(f"Simple model: ±{avg_error_simple:.1f} M tons")
print(f"Full model: ±{avg_error_full:.1f} M tons")
print(f"Improvement: {avg_error_simple - avg_error_full:.1f} M tons less error")

# Compare to average CO2
avg_co2 = np.mean(co2)
print(f"\nAverage CO2: {avg_co2:.1f} M tons")
print(f"Simple model error: {avg_error_simple/avg_co2*100:.1f}% of average")
print(f"Full model error: {avg_error_full/avg_co2*100:.1f}% of average")

print("\n" + "="*60)
print("CONCLUSIONS")
print("="*60)
print(f"\n1. Year alone gives LSE of {lse1:,.0f}")
print(f"2. Adding temperature reduces LSE by {lse1 - lse2:,.0f}")
print(f"3. Adding rainfall reduces further by {lse2 - lse3:,.0f}")
print(f"4. Adding population reduces by {lse3 - lse4:,.0f}")
print(f"\n5. Full model has lowest LSE: {lse4:,.0f}")
print(f"6. Best fit equation: CO2 = {beta4[0]:.0f} + {beta4[1]:.3f}×Year + {beta4[2]:.3f}×Temp + {beta4[3]:.3f}×Rain + {beta4[4]:.8f}×Pop")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


climateData = pd.read_csv("ClimateDataset.csv")
numeric_data = climateData.select_dtypes(include=[np.number])


climateData.plot(y='Rainfall', x="Year")
plt.savefig(f"saved_figures/rainfall_mean.png")
plt.clf()

climateData.plot(y='CO2 emissions', x="Year")
plt.savefig(f"saved_figures/CO2byYear.png")
plt.clf()

climateData.plot(y='Temperature', x="Year")
plt.savefig(f"saved_figures/CO2byYear.png")
plt.clf()


# Simple analysis: Temperature trend over time
if 'Year' in numeric_data.columns and 'Temperature Mean ()' in numeric_data.columns:
    X = numeric_data[['Year']]
    y = numeric_data['Temperature Mean ()']
    
    model = LinearRegression()
    model.fit(X, y)
    
    print(f"Temperature increase per year: {model.coef_[0]:.4f}°C/year")
    print(f"Predicted temp in 2050: {model.predict([[2050]])[0]:.2f}°C")
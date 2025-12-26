import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


climateData = pd.read_csv("ClimateDataset.csv")
numeric_data = climateData.select_dtypes(include=[np.number])


climateData.plot(y='Rainfall', x="Year")
plt.savefig(f"saved_figures/rainfallByYear.png")
plt.clf()

climateData.plot(y='CO2 emissions', x="Year")
plt.savefig(f"saved_figures/CO2byYear.png")
plt.clf()

climateData.plot(y='Temperature', x="Year")
plt.savefig(f"saved_figures/TempbyYear.png")
plt.clf()


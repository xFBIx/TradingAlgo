from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# Download data
ticker = "IGL.NS"
data = yf.download(ticker, start="2024-01-01", end="2025-01-21")
data = data[["Open", "High", "Low", "Close"]]

# Identify local maxima and minima
high_indices = argrelextrema(data["High"].values, np.greater, order=10)[0]
low_indices = argrelextrema(data["Low"].values, np.less, order=10)[0]

# Initialize columns for high and low points
data["High_Point"] = np.nan
data["Low_Point"] = np.nan

# Use numeric indices to assign high and low points
data.iloc[high_indices, data.columns.get_loc("High_Point")] = data.iloc[high_indices][
    "High"
]
data.iloc[low_indices, data.columns.get_loc("Low_Point")] = data.iloc[low_indices][
    "Low"
]

# Plot with high and low points
plt.figure(figsize=(14, 8))

# Plot close price
plt.plot(data["Close"], label="Close Price", alpha=0.5)

# Plot support and resistance levels
data["Support"] = data["Low"].rolling(window=20, center=True).min()
data["Resistance"] = data["High"].rolling(window=20, center=True).max()
plt.plot(data["Support"], label="Support Level", linestyle="--", color="green")
plt.plot(data["Resistance"], label="Resistance Level", linestyle="--", color="red")

# Plot high and low points
plt.scatter(
    data.index, data["High_Point"], label="High Points", marker="^", color="blue"
)
plt.scatter(
    data.index, data["Low_Point"], label="Low Points", marker="v", color="orange"
)

# Add legend and title
plt.legend()
plt.title("Support, Resistance, and High/Low Points")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid()
plt.show()

import pandas as pd
import mplfinance as mpf
import numpy as np
from data_manupulation import clean_csv

# Load and clean CSV file
csv_file = "GPPL.csv"
stock_name = csv_file.split(".")[0]
clean_csv(csv_file, csv_file)
data = pd.read_csv(csv_file, index_col="Date", parse_dates=True)
data.sort_index(ascending=True, inplace=True)


# Calculate RSI
def calculate_rsi(data, period=14):
    delta = data["Close"].diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gain = gains.rolling(window=period, min_periods=1).mean()
    avg_loss = losses.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


data["RSI"] = calculate_rsi(data)

# Calculate Support and Resistance Levels

# This part should be done with the help of ML to help choose the correct window size and points.
"""
Rolling Window: A rolling window of specified days (30 days in this case) moves through the data series. For each position of the window, it calculates the minimum and maximum values within that window. This method helps in smoothing out short-term fluctuations and highlights more significant price levels that hold over time.
Local Minima: These are points where a price is lower than both the prices before and after it in the window. They potentially indicate support levels.
Local Maxima: These are points where a price is higher than both the prices before and after it in the window. They potentially indicate resistance levels.

rolling(window=window, center=True).min() and max(): These functions compute the minimum and maximum values, respectively, within a centered rolling window. By setting center=True, the window is centered over each data point, using data from both before and after the point to calculate minima and maxima. This helps in evenly distributing the look-back and look-forward periods, which is crucial for accurately identifying significant levels.
"""


def find_levels(data, window=10):
    # Calculate rolling minimum and maximum
    minima = data["Low"].rolling(window=window, center=True).min()
    maxima = data["High"].rolling(window=window, center=True).max()
    # print(data["Low"])
    return minima, maxima


support_levels, resistance_levels = find_levels(data)

# Create RSI subplot
rsi_plot = mpf.make_addplot(data["RSI"], panel=2, ylabel="RSI")

# Create support and resistance lines as scatter plots
support_lines = mpf.make_addplot(
    np.where(data["Low"] == support_levels, support_levels, np.nan),
    type="scatter",
    markersize=50,
    marker="o",
    color="green",
    ylabel="Support",
)
resistance_lines = mpf.make_addplot(
    np.where(data["High"] == resistance_levels, resistance_levels, np.nan),
    type="scatter",
    markersize=50,
    marker="o",
    color="red",
    ylabel="Resistance",
)

# Plot everything together
# Merge the additional plots correctly into a single list before passing to addplot
additional_plots = [rsi_plot, support_lines, resistance_lines]

mpf.plot(
    data,
    type="candle",
    style="yahoo",
    title=stock_name,
    volume=True,
    addplot=additional_plots,
    panel_ratios=(3, 1, 1),  # Ratio of main chart, volume, and RSI chart
)

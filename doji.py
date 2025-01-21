import pandas as pd
import mplfinance as mpf
import numpy as np
from data_manupulation import clean_csv

# Load and clean CSV file
csv_file = "./csv_data/TATSTE.csv"
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
def find_levels(data, window=10):
    minima = data["Low"].rolling(window=window, center=True).min()
    maxima = data["High"].rolling(window=window, center=True).max()
    return minima, maxima


support_levels, resistance_levels = find_levels(data)


# Identify Doji candlesticks (with close within 2% of open)
def identify_dojis(data):
    # Check if close price is within 2% of open price
    doji_condition = (data["Close"] >= data["Open"] * 0.997) & (
        data["Close"] <= data["Open"] * 1.003
    )
    return doji_condition


doji_days = identify_dojis(data)

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

# Mark Doji days with a blue star
doji_lines = mpf.make_addplot(
    np.where(doji_days, data["Close"], np.nan),
    type="scatter",
    markersize=100,
    marker="*",
    color="blue",
    ylabel="Doji",
)

# Plot everything together
additional_plots = [rsi_plot, support_lines, resistance_lines, doji_lines]

mpf.plot(
    data,
    type="candle",
    style="yahoo",
    title=stock_name,
    volume=True,
    addplot=additional_plots,
    panel_ratios=(3, 1, 1),  # Ratio of main chart, volume, and RSI chart
)

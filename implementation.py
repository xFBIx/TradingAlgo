import pandas as pd
import matplotlib.pyplot as plt

from supp_resis_bands import calculate_rsi

data = pd.read_csv("./csv_data/GPPL.csv", index_col="Date", parse_dates=True)
data.sort_index(ascending=True, inplace=True)

# Calculate indicators
data["RSI"] = calculate_rsi(data)

# Assuming the RSI values are in a column named 'RSI'
rsi_series = data["RSI"]

# Create a figure and axis for plotting
plt.figure(figsize=(12, 6))
plt.plot(rsi_series.index, rsi_series, label="RSI", color="blue")

# Identify points where RSI is below 30
below_30 = rsi_series < 30

# Track subsequent numbers and draw a vertical line when RSI goes above 30
for i in range(1, len(rsi_series)):
    if below_30[i - 1] and rsi_series[i] > 30:
        plt.axvline(x=rsi_series.index[i], color="red", linestyle="--", linewidth=0.5)


# Function to track drops of 20 or more within 7 consecutive values
def track_drops_and_increases(series):
    for i in range(len(series) - 7):
        # Check for a drop of 20 or more within the next 7 values
        if series[i] - series[i + 7] >= 20:
            # Now track subsequent entries to find when it starts increasing again
            for j in range(i + 7, len(series)):
                if series[j] > series[j - 1]:  # Check if the value starts increasing
                    plt.axvline(
                        x=series.index[j], color="green", linestyle="--", linewidth=0.5
                    )
                    break  # Stop tracking after the first increase


# Call the function to track drops and increases
track_drops_and_increases(rsi_series)

# Add a horizontal line at RSI = 30 for reference
plt.axhline(y=30, color="red", linestyle="--", label="RSI = 30")

# Add labels and title
plt.title("RSI with Vertical Lines for Crossings and Increases")
plt.xlabel("Date")
plt.ylabel("RSI")
plt.legend()
plt.grid()

# Show the plot
plt.show()

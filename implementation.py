import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from supp_resis_bands import calculate_rsi

# Load and process data using yfinance
ticker = yf.Ticker("GPPL.NS")
stock_name = ticker.info["longName"]
data = ticker.history(start="2023-01-01", end="2024-01-01", interval="1d")

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
    if below_30[i - 1]:
        # Check if subsequent values remain within the range of 25.0 to 35.0
        if all(25.0 <= rsi_series[j] <= 35.0 for j in range(i, len(rsi_series))):
            continue  # Skip drawing if all subsequent values are within the range
        else:
            # Track until we find a value greater than 35
            for j in range(i, len(rsi_series)):
                if rsi_series[j] > 35:
                    # Draw a red line on the last day that has the float value greater than 35
                    plt.axvline(
                        x=rsi_series.index[j - 1],
                        color="red",
                        linestyle="--",
                        linewidth=0.5,
                    )
                    break  # Stop tracking after the first increase


# Function to track drops of 20 or more within 7 consecutive values
def track_drops_and_rsi(series):
    green_line_drawn = False  # State variable to track if a green line has been drawn

    for i in range(len(series) - 7):
        # Check for a drop of 20 or more within the next 7 values
        if series[i] - series[i + 7] >= 20:
            # Only track RSI if a green line has not been drawn
            if not green_line_drawn:
                previous_rsi = series[i + 7]
                for j in range(i + 8, len(series)):
                    current_rsi = series[j]

                    # Check if the current RSI is more than 5 greater than the previous RSI
                    if current_rsi > previous_rsi + 5:
                        plt.axvline(
                            x=series.index[j],
                            color="green",
                            linestyle="--",
                            linewidth=0.5,
                        )
                        green_line_drawn = True  # Set the state variable to True
                        break  # Stop tracking after the first valid increase

                    # If we don't have data for the next value, we can also print a line
                    if j == len(series) - 1:
                        plt.axvline(
                            x=series.index[j],
                            color="green",
                            linestyle="--",
                            linewidth=0.5,
                        )
                        green_line_drawn = True  # Set the state variable to True
                        break  # Stop tracking after the last value

                    # If the increase is within 5, we ignore it
                    previous_rsi = (
                        current_rsi  # Update previous_rsi for the next iteration
                    )

        # Reset the state variable if a new drop is detected after a green line was drawn
        if green_line_drawn and series[i] - series[i + 7] < 20:
            green_line_drawn = False  # Reset to allow tracking again


# Call the function to track drops and increases
track_drops_and_rsi(rsi_series)

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

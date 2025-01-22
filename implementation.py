import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from supp_resis_bands import calculate_rsi
from datetime import datetime


def analyze_rsi_triggers(symbol, start_date, end_date, interval="1d"):
    """
    Analyze RSI triggers for oversold conditions and significant drops

    Parameters:
    symbol (str): Stock symbol with exchange (e.g., 'GPPL.NS')
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    interval (str): Data interval

    Returns:
    tuple: Set of trigger dates and DataFrame with RSI values
    """
    # Load and process data using yfinance
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date, interval=interval)

    # Calculate RSI
    data["RSI"] = calculate_rsi(data)
    rsi_series = data["RSI"]

    # Use a set to store unique trigger dates
    trigger_dates = set()

    def check_oversold_condition():
        """Check for RSI below 30 and subsequent movement"""
        below_30 = rsi_series < 30

        for i in range(1, len(rsi_series)):
            if below_30[i - 1]:
                # Skip if in consolidation range (25-35)
                subsequent_values = rsi_series[i:]
                if all(25.0 <= val <= 35.0 for val in subsequent_values):
                    continue

                # Find first value above 35
                for j in range(i, len(rsi_series)):
                    if rsi_series[j] > 35:
                        trigger_date = rsi_series.index[j - 1]
                        trigger_dates.add(trigger_date)
                        break

    def check_sharp_drops():
        """Check for drops of 20 or more within 7 consecutive values"""
        processed_drops = set()  # Track processed drop periods

        for i in range(len(rsi_series) - 7):
            drop_end_idx = i + 7

            # Check if this period has been processed
            if drop_end_idx in processed_drops:
                continue

            # Check for drop of 20 or more
            if rsi_series[i] - rsi_series[drop_end_idx] >= 20:
                previous_rsi = rsi_series[drop_end_idx]

                # Mark this drop period as processed
                processed_drops.add(drop_end_idx)

                # Look for recovery (increase of more than 5)
                for j in range(drop_end_idx + 1, len(rsi_series)):
                    current_rsi = rsi_series[j]

                    if current_rsi > previous_rsi + 5:
                        trigger_dates.add(rsi_series.index[j])
                        break

                    # Handle end of data
                    if j == len(rsi_series) - 1:
                        trigger_dates.add(rsi_series.index[j])
                        break

                    previous_rsi = current_rsi

    # Run both analyses
    check_oversold_condition()
    check_sharp_drops()

    # Convert dates to datetime for consistent formatting
    trigger_dates = {pd.to_datetime(date) for date in trigger_dates}

    # Sort trigger dates for better readability
    sorted_trigger_dates = sorted(trigger_dates)

    return sorted_trigger_dates, data


# Example usage
if __name__ == "__main__":
    trigger_dates, data = analyze_rsi_triggers(
        symbol="GPPL.NS", start_date="2024-01-01", end_date="2025-01-22"
    )

    print("\nUnique Trigger Dates:")
    for date in trigger_dates:
        print(f"- {date.strftime('%Y-%m-%d')}")

    print(f"\nTotal unique triggers found: {len(trigger_dates)}")

    # Optional: Print RSI values on trigger dates
    # print("\nRSI values on trigger dates:")
    # for date in trigger_dates:
    #     rsi_value = data.loc[date, "RSI"]
    #     print(f"- {date.strftime('%Y-%m-%d')}: RSI = {rsi_value:.2f}")

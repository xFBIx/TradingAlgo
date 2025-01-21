import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import linregress
from typing import List, Tuple, Dict
from datetime import timedelta


class SupportResistanceTrader:
    def __init__(
        self,
        window_size: int = 5,
        num_points: int = 2,
        threshold: float = 0.015,
        trend_window: int = 20,
        min_trend_points: int = 3,
    ):
        self.window_size = window_size
        self.num_points = num_points
        self.threshold = threshold
        self.trend_window = trend_window
        self.min_trend_points = min_trend_points

    def find_local_mins_maxs(
        self, data: pd.DataFrame
    ) -> Tuple[List[float], List[float]]:
        """Find local minima and maxima in price data."""
        highs = data["High"].values
        lows = data["Low"].values

        resistance_points = []
        support_points = []

        for i in range(self.window_size, len(highs) - self.window_size):
            if highs[i] == max(highs[i - self.window_size : i + self.window_size + 1]):
                resistance_points.append((i, highs[i]))

        for i in range(self.window_size, len(lows) - self.window_size):
            if lows[i] == min(lows[i - self.window_size : i + self.window_size + 1]):
                support_points.append((i, lows[i]))

        return support_points, resistance_points

    def find_trend_lines(
        self, data: pd.DataFrame, extremes: List[Tuple[int, float]]
    ) -> List[Tuple[float, float]]:
        """Find trend lines using linear regression on price extremes."""
        trend_lines = []

        for i in range(len(extremes) - self.min_trend_points + 1):
            for j in range(i + self.min_trend_points - 1, len(extremes)):
                points = extremes[i : j + 1]
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]

                # Perform linear regression
                slope, intercept, r_value, _, _ = linregress(x_coords, y_coords)

                # Check if the trend line is significant (R² > 0.8)
                if r_value**2 > 0.8:
                    trend_lines.append((slope, intercept))

        return trend_lines

    def identify_vertical_bands(self, data: pd.DataFrame) -> List[str]:
        """Identify significant vertical time bands based on volume and price movement."""
        significant_dates = []

        # Calculate daily returns and volume changes
        data["Returns"] = data["Close"].pct_change()
        data["VolChange"] = data["Volume"].pct_change()

        # Find days with significant price movements and volume
        for date in data.index[1:]:
            if (
                abs(data.loc[date, "Returns"]) > 0.02  # 2% price movement
                and data.loc[date, "VolChange"] > 0.5
            ):  # 50% volume increase
                significant_dates.append(date)

        return significant_dates

    def plot_analysis(
        self,
        data: pd.DataFrame,
        support_levels: List[float],
        resistance_levels: List[float],
        trend_lines: Dict[str, List[Tuple[float, float]]],
        vertical_bands: List[str],
    ):
        """Plot the stock data with horizontal levels, trend lines, and vertical bands."""
        plt.figure(figsize=(15, 7))

        # Plot stock data
        plt.plot(
            data.index, data["Close"], label="Close Price", color="black", alpha=0.7
        )
        plt.fill_between(data.index, data["High"], data["Low"], color="gray", alpha=0.2)

        # Plot horizontal support/resistance levels
        for i, level in enumerate(support_levels):
            plt.axhline(
                y=level,
                color="g",
                linestyle="--",
                alpha=0.7,
                label=f"Support {i+1}: {level:.2f}",
            )

        for i, level in enumerate(resistance_levels):
            plt.axhline(
                y=level,
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"Resistance {i+1}: {level:.2f}",
            )

        # Plot trend lines
        x_range = np.arange(len(data))
        for slope, intercept in trend_lines["support"]:
            y_values = slope * x_range + intercept
            plt.plot(data.index, y_values, "g-", alpha=0.5, label="Support Trend")

        for slope, intercept in trend_lines["resistance"]:
            y_values = slope * x_range + intercept
            plt.plot(data.index, y_values, "r-", alpha=0.5, label="Resistance Trend")

        # Plot vertical bands
        for date in vertical_bands:
            plt.axvline(x=date, color="blue", alpha=0.2)

        plt.title("Stock Price Analysis with Trends and Bands")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def analyze_stock(self, data: pd.DataFrame):
        """Complete analysis including horizontal levels, trend lines, and vertical bands."""
        # Find horizontal levels
        support_points, resistance_points = self.find_local_mins_maxs(data)

        # Find trend lines
        support_trends = self.find_trend_lines(data, support_points)
        resistance_trends = self.find_trend_lines(data, resistance_points)

        # Find vertical bands
        vertical_bands = self.identify_vertical_bands(data)

        # Convert points to levels for horizontal S/R
        price_range = data["High"].max() - data["Low"].min()
        support_levels = self._cluster_levels(
            [p[1] for p in support_points], price_range
        )
        resistance_levels = self._cluster_levels(
            [p[1] for p in resistance_points], price_range
        )

        trend_lines = {"support": support_trends, "resistance": resistance_trends}

        return support_levels, resistance_levels, trend_lines, vertical_bands

    def _cluster_levels(self, points: List[float], price_range: float) -> List[float]:
        """Cluster nearby price levels."""
        if not points:
            return []

        adaptive_threshold = self.threshold * price_range
        points = sorted(points)
        clusters = []
        current_cluster = [points[0]]

        for point in points[1:]:
            if abs(point - np.mean(current_cluster)) <= adaptive_threshold:
                current_cluster.append(point)
            else:
                if len(current_cluster) >= self.num_points:
                    clusters.append(np.mean(current_cluster))
                current_cluster = [point]

        if len(current_cluster) >= self.num_points:
            clusters.append(np.mean(current_cluster))

        return clusters


def main():
    # Download stock data
    symbol = "AAPL"
    stock = yf.Ticker(symbol)
    data = stock.history(period="12mo")

    # Initialize and run the trader
    trader = SupportResistanceTrader(
        window_size=5,
        num_points=2,
        threshold=0.015,
        trend_window=20,
        min_trend_points=3,
    )

    # Perform complete analysis
    support_levels, resistance_levels, trend_lines, vertical_bands = (
        trader.analyze_stock(data)
    )

    print(f"\nAnalyzing {symbol} stock:")
    print("\nHorizontal Support Levels:", [f"{level:.2f}" for level in support_levels])
    print(
        "Horizontal Resistance Levels:", [f"{level:.2f}" for level in resistance_levels]
    )
    print(f"\nFound {len(trend_lines['support'])} support trends")
    print(f"Found {len(trend_lines['resistance'])} resistance trends")
    print(f"Found {len(vertical_bands)} significant vertical bands")

    # Plot the analysis
    trader.plot_analysis(
        data, support_levels, resistance_levels, trend_lines, vertical_bands
    )


if __name__ == "__main__":
    main()

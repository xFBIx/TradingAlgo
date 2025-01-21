import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from typing import List, Tuple


class SupportResistanceTrader:
    def __init__(
        self, window_size: int = 5, num_points: int = 2, threshold: float = 0.015
    ):
        self.window_size = window_size
        self.num_points = num_points
        self.threshold = threshold

    def find_local_mins_maxs(
        self, data: pd.DataFrame
    ) -> Tuple[List[float], List[float]]:
        """
        Find local minima and maxima in price data using a more sensitive approach.
        """
        highs = data["High"].values
        lows = data["Low"].values

        resistance_points = []
        support_points = []

        # Find local maxima
        for i in range(self.window_size, len(highs) - self.window_size):
            if highs[i] == max(highs[i - self.window_size : i + self.window_size + 1]):
                resistance_points.append(highs[i])

        # Find local minima
        for i in range(self.window_size, len(lows) - self.window_size):
            if lows[i] == min(lows[i - self.window_size : i + self.window_size + 1]):
                support_points.append(lows[i])

        return support_points, resistance_points

    def _cluster_levels(self, points: List[float], price_range: float) -> List[float]:
        """
        Cluster nearby price levels with adaptive threshold based on price range.
        """
        if not points:
            return []

        # Adjust threshold based on price range
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

    def identify_support_resistance(
        self, data: pd.DataFrame
    ) -> Tuple[List[float], List[float]]:
        """
        Identify support and resistance levels with improved sensitivity.
        """
        # Calculate price range for adaptive thresholding
        price_range = data["High"].max() - data["Low"].min()

        # Find local minimums and maximums
        support_points, resistance_points = self.find_local_mins_maxs(data)

        # Cluster levels with adaptive threshold
        resistance_levels = self._cluster_levels(resistance_points, price_range)
        support_levels = self._cluster_levels(support_points, price_range)

        return support_levels, resistance_levels

    def plot_analysis(
        self,
        data: pd.DataFrame,
        support_levels: List[float],
        resistance_levels: List[float],
    ):
        """
        Plot the stock data with support and resistance levels.
        """
        plt.figure(figsize=(15, 7))

        # Plot stock data
        plt.plot(
            data.index, data["Close"], label="Close Price", color="black", alpha=0.7
        )
        plt.fill_between(data.index, data["High"], data["Low"], color="gray", alpha=0.2)

        # Plot support levels
        for i, level in enumerate(support_levels):
            plt.axhline(
                y=level,
                color="g",
                linestyle="--",
                alpha=0.7,
                label=f"Support {i+1}: {level:.2f}",
            )

        # Plot resistance levels
        for i, level in enumerate(resistance_levels):
            plt.axhline(
                y=level,
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"Resistance {i+1}: {level:.2f}",
            )

        plt.title("Stock Price with Support and Resistance Levels")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def main():
    # Download stock data
    symbol = "IGL.NS"
    stock = yf.Ticker(symbol)
    data = stock.history(period="12mo")

    # Initialize and run the trader with more sensitive parameters
    trader = SupportResistanceTrader(
        window_size=5,  # Smaller window for more frequent level detection
        num_points=3,  # Fewer points needed to confirm a level
        threshold=0.035,  # More sensitive threshold
    )

    support_levels, resistance_levels = trader.identify_support_resistance(data)

    print(f"\nAnalyzing {symbol} stock:")
    print("\nSupport Levels:", [f"{level:.2f}" for level in support_levels])
    print("Resistance Levels:", [f"{level:.2f}" for level in resistance_levels])

    # Plot the analysis
    trader.plot_analysis(data, support_levels, resistance_levels)


if __name__ == "__main__":
    main()

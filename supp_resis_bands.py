import pandas as pd
import mplfinance as mpf
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from itertools import combinations


def calculate_rsi(data, period=14):
    delta = data["Close"].diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gain = gains.rolling(window=period, min_periods=1).mean()
    avg_loss = losses.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def find_levels(data, window=10):
    minima = data["Low"].rolling(window=window, center=True).min()
    maxima = data["High"].rolling(window=window, center=True).max()
    return minima, maxima


def filter_significant_points(points, price_threshold=0.02, time_threshold=5):
    """
    Filter points to keep only significant ones based on price and time differences
    """
    if not points:
        return []

    filtered = [points[0]]
    for point in points[1:]:
        last_point = filtered[-1]
        price_diff = abs(point[1] - last_point[1]) / last_point[1]
        time_diff = point[0] - last_point[0]

        if price_diff > price_threshold and time_diff > time_threshold:
            filtered.append(point)

    return filtered


def find_support_lines(
    x_values, y_values, min_points=3, r2_threshold=0.95, min_slope=0.001
):
    """
    Find trend lines with improved filtering and validation
    """
    # Convert inputs to numpy arrays and remove NaN values
    valid_points = [(x, y) for x, y in zip(x_values, y_values) if not np.isnan(y)]
    if len(valid_points) < min_points:
        return []

    x_values = np.array([p[0] for p in valid_points]).reshape(-1, 1)
    y_values = np.array([p[1] for p in valid_points])

    points = list(zip(x_values.flatten(), y_values))
    lines = []

    for i in range(len(points) - min_points + 1):
        for j in range(i + min_points, len(points) + 1):
            segment = points[i:j]
            x = np.array([p[0] for p in segment]).reshape(-1, 1)
            y = np.array([p[1] for p in segment])

            if len(x) < min_points:
                continue

            model = LinearRegression()
            model.fit(x, y)
            r2 = model.score(x, y)
            slope = abs(model.coef_[0])

            if r2 > r2_threshold and slope > min_slope:
                # Calculate the predicted y values
                y_pred = model.predict(x)

                # Calculate the perpendicular distances
                distances = np.abs(y - y_pred)  # Distance from points to the line
                variance = np.var(distances)

                # print(f"Variance for segment {segment}: {variance}")

                # If variance exceeds 5, filter points
                while variance > 5:
                    # Identify the point with the maximum distance above the line
                    above_line_points = [
                        (p[0], p[1]) for p in segment if p[1] > y_pred[segment.index(p)]
                    ]
                    if not above_line_points:
                        break  # No points above the line to remove

                    # Find the point with the maximum distance
                    farthest_point = max(
                        above_line_points,
                        key=lambda p: abs(p[1] - y_pred[segment.index(p)]),
                    )
                    segment.remove(farthest_point)

                    # Recalculate x and y for the new segment
                    x = np.array([p[0] for p in segment]).reshape(-1, 1)
                    y = np.array([p[1] for p in segment])

                    if len(x) < min_points:
                        break

                    model.fit(x, y)
                    y_pred = model.predict(x)
                    distances = np.abs(y - y_pred)
                    variance = np.var(distances)

                # After filtering, check if we still have enough points
                if len(segment) >= min_points:
                    lines.append((segment, model, r2))

    # Filter overlapping lines
    if not lines:
        return []

    lines.sort(key=lambda x: (len(x[0]), x[2]), reverse=True)
    filtered_lines = []
    used_points = set()

    for line, model, r2 in lines:
        points_set = set(point[0] for point in line)
        if not points_set.intersection(used_points):
            filtered_lines.append((line, model))
            used_points.update(points_set)

    return filtered_lines


def find_resistance_lines(
    x_values, y_values, min_points=3, r2_threshold=0.95, min_slope=0.001
):
    """
    Find trend lines with improved filtering and validation
    """
    # Convert inputs to numpy arrays and remove NaN values
    valid_points = [(x, y) for x, y in zip(x_values, y_values) if not np.isnan(y)]
    if len(valid_points) < min_points:
        return []

    x_values = np.array([p[0] for p in valid_points]).reshape(-1, 1)
    y_values = np.array([p[1] for p in valid_points])

    points = list(zip(x_values.flatten(), y_values))
    lines = []

    for i in range(len(points) - min_points + 1):
        for j in range(i + min_points, len(points) + 1):
            segment = points[i:j]
            x = np.array([p[0] for p in segment]).reshape(-1, 1)
            y = np.array([p[1] for p in segment])

            if len(x) < min_points:
                continue

            model = LinearRegression()
            model.fit(x, y)
            r2 = model.score(x, y)
            slope = abs(model.coef_[0])

            if r2 > r2_threshold and slope > min_slope:
                # Calculate the predicted y values
                y_pred = model.predict(x)

                # Calculate the perpendicular distances
                distances = np.abs(y - y_pred)  # Distance from points to the line
                variance = np.var(distances)

                # print(f"Variance for segment {segment}: {variance}")

                # If variance exceeds 5, filter points
                while variance > 5:
                    # Identify the point with the maximum distance below the line
                    below_line_points = [
                        (p[0], p[1]) for p in segment if p[1] < y_pred[segment.index(p)]
                    ]
                    if not below_line_points:
                        break  # No points below the line to remove

                    # Find the point with the maximum distance
                    farthest_point = max(
                        below_line_points,
                        key=lambda p: abs(p[1] - y_pred[segment.index(p)]),
                    )
                    segment.remove(farthest_point)

                    # Recalculate x and y for the new segment
                    x = np.array([p[0] for p in segment]).reshape(-1, 1)
                    y = np.array([p[1] for p in segment])

                    if len(x) < min_points:
                        break

                    model.fit(x, y)
                    y_pred = model.predict(x)
                    distances = np.abs(y - y_pred)
                    variance = np.var(distances)

                # After filtering, check if we still have enough points
                if len(segment) >= min_points:
                    lines.append((segment, model, r2))

    # Filter overlapping lines
    if not lines:
        return []

    lines.sort(key=lambda x: (len(x[0]), x[2]), reverse=True)
    filtered_lines = []
    used_points = set()

    for line, model, r2 in lines:
        points_set = set(point[0] for point in line)
        if not points_set.intersection(used_points):
            filtered_lines.append((line, model))
            used_points.update(points_set)

    return filtered_lines


def create_plot_data(data, trend_lines, is_support=True):
    """
    Create plot data for trend lines with validation
    """
    plot_data = pd.Series(np.nan, index=data.index)

    if not trend_lines:
        return None

    valid_data = False
    for line_points, model in trend_lines:
        if len(line_points) < 2:
            continue

        start_idx = int(line_points[0][0])
        end_idx = int(line_points[-1][0])

        if start_idx >= len(data) or end_idx >= len(data):
            continue

        x_predict = np.array(range(start_idx, end_idx + 1)).reshape(-1, 1)
        y_predict = model.predict(x_predict)

        plot_data.iloc[start_idx : end_idx + 1] = y_predict
        valid_data = True

    return plot_data if valid_data else None


def main(symbol, start_date, end_date, interval="1d"):

    # Load and process data using yfinance
    ticker = yf.Ticker(symbol)
    stock_name = ticker.info["longName"]
    data = ticker.history(start=start_date, end=end_date, interval=interval)

    # Calculate indicators
    data["RSI"] = calculate_rsi(data)
    support_levels, resistance_levels = find_levels(data)
    # Create base plots
    additional_plots = []

    # Add RSI plot
    rsi_plot = mpf.make_addplot(data["RSI"], panel=2, ylabel="RSI")
    additional_plots.append(rsi_plot)

    # Add scatter plots only if valid points exist
    support_points = [
        (i, level)
        for i, level in enumerate(support_levels)
        if not np.isnan(level) and data["Low"].iloc[i] == level
    ]

    resistance_points = [
        (i, level)
        for i, level in enumerate(resistance_levels)
        if not np.isnan(level) and data["High"].iloc[i] == level
    ]

    if support_points:
        support_scatter = mpf.make_addplot(
            np.where(data["Low"] == support_levels, support_levels, np.nan),
            type="scatter",
            markersize=50,
            marker="o",
            color="green",
            ylabel="Support",
        )
        additional_plots.append(support_scatter)

    if resistance_points:
        resistance_scatter = mpf.make_addplot(
            np.where(data["High"] == resistance_levels, resistance_levels, np.nan),
            type="scatter",
            markersize=50,
            marker="o",
            color="red",
            ylabel="Resistance",
        )
        additional_plots.append(resistance_scatter)

    # Find and add trend lines only if enough points exist
    if len(support_points) >= 3:
        support_lines = find_support_lines(
            [p[0] for p in support_points],
            [p[1] for p in support_points],
            min_points=3,
            r2_threshold=0.90,
            min_slope=0.0001,
        )

        support_trend_data = create_plot_data(data, support_lines)
        if support_trend_data is not None and not support_trend_data.isna().all():
            support_trend = mpf.make_addplot(
                support_trend_data, color="green", width=1, alpha=0.7
            )
            additional_plots.append(support_trend)

    if len(resistance_points) >= 3:
        resistance_lines = find_resistance_lines(
            [p[0] for p in resistance_points],
            [p[1] for p in resistance_points],
            min_points=3,
            r2_threshold=0.90,
            min_slope=0.0001,
        )

        resistance_trend_data = create_plot_data(data, resistance_lines)
        if resistance_trend_data is not None and not resistance_trend_data.isna().all():
            resistance_trend = mpf.make_addplot(
                resistance_trend_data, color="red", width=1, alpha=0.7
            )
            additional_plots.append(resistance_trend)

    # Create final plot only if we have valid plots to add
    if additional_plots:
        mpf.plot(
            data,
            type="candle",
            style="yahoo",
            title=stock_name,
            volume=True,
            addplot=additional_plots,
            panel_ratios=(3, 1, 1),
        )


# Run the analysis
if __name__ == "__main__":
    main(
        symbol="GPPL.NS", start_date="2023-01-01", end_date="2024-01-01", interval="1d"
    )

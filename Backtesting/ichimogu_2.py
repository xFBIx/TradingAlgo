import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import copy
import matplotlib.pyplot as plt
import warnings
from functions import *

warnings.simplefilter(action="ignore", category=FutureWarning)

# Download historical data (monthly) for DJI constituent stocks

# tickers = ["LXCHEM.NS", "GPPL.NS"]
tickers = [
    # "ITC.NS",
    "JSWENERGY.NS",
    # "LICHSGFIN.NS",
    # "ASHOKLEY.NS",
    # "IDEA.NS",
    # "SUZLON.NS",
    # "EASEMYTRIP.NS",
    # "IRB.NS",
    # "YESBANK.NS",
    # "IDFCFIRSTB.NS",
    # "PNB.NS",
    "BHEL.NS",
    # "SAIL.NS",
    # "NATIONALUM.NS",
    "BANKBARODA.NS",
    # "JINDALSTEL.NS",
    # "TATACHEM.NS",
    "SUNTV.NS",
    # "M&MFIN.NS",
    # "BANKINDIA.NS",
]

ohlc_mon = {}  # directory with ohlc value for each stock
# start_date = dt.datetime.today() - dt.timedelta(365)
# end_date = dt.datetime.today()
start_date = "2022-01-01"
end_date = "2025-01-01"

# looping over tickers and creating a dataframe with close prices
for ticker in tickers:
    ohlc_mon[ticker] = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        multi_level_index=False,
    )
    ohlc_mon[ticker].dropna(inplace=True, how="all")

tickers = (
    ohlc_mon.keys()
)  # redefine tickers variable after removing any tickers with corrupted data

################################Backtesting####################################

# calculating monthly return for each stock and consolidating return info by stock in a separate dataframe
ohlc_dict = copy.deepcopy(ohlc_mon)
tickers_signal = {}
tickers_ret = {}
for ticker in tickers:
    print("calculating Ichimogu Values", ticker)
    ohlc_dict[ticker]["ATR"] = ATR(ohlc_dict[ticker], 20)
    ohlc_dict[ticker]["RSI"] = RSI(ohlc_dict[ticker], 14)
    ohlc_dict[ticker][["Conversion", "Base", "Leading_a", "Leading_b", "Lagging"]] = (
        ichimogu(ohlc_dict[ticker], 9, 26, 52, 26)
    )
    # ohlc_dict[ticker].dropna(inplace=True)
    tickers_signal[ticker] = ""
    tickers_ret[ticker] = [0]


# identifying signals and calculating daily return
for ticker in tickers:
    print("calculating returns for ", ticker)

    for i in range(1, len(ohlc_dict[ticker])):
        conversion = ohlc_dict[ticker]["Conversion"].iloc[i]
        base = ohlc_dict[ticker]["Base"].iloc[i]
        Leading_a = ohlc_dict[ticker]["Leading_a"].iloc[i]
        Leading_b = ohlc_dict[ticker]["Leading_b"].iloc[i]
        x = conversion - base
        y = Leading_a - Leading_b
        z = (conversion + base / 2) - (Leading_a + Leading_b / 2)

        if tickers_signal[ticker] == "":
            tickers_ret[ticker].append(0)
            if (
                x > 0
                and y > 0
                and x > 0.05 * base
                and y > 0.001 * Leading_b
                and min(
                    ohlc_dict[ticker]["Open"].iloc[i],
                    ohlc_dict[ticker]["Close"].iloc[i],
                )
                > Leading_a
            ):
                tickers_signal[ticker] = "Buy"

        elif tickers_signal[ticker] == "Buy":
            if (
                max(
                    ohlc_dict[ticker]["Open"].iloc[i - 1],
                    ohlc_dict[ticker]["Close"].iloc[i - 1],
                )
                < min(
                    ohlc_dict[ticker]["Leading_a"].iloc[i - 1],
                    ohlc_dict[ticker]["Leading_b"].iloc[i - 1],
                )
                and max(
                    ohlc_dict[ticker]["Open"].iloc[i],
                    ohlc_dict[ticker]["Close"].iloc[i],
                )
                < min(Leading_a, Leading_b)
            ) or (y < 0 and -y > 0.02 * Leading_a):
                tickers_signal[ticker] = (
                    ""  # Reset signal to wait for next buy opportunity
                )
                tickers_ret[ticker].append(
                    (
                        ohlc_dict[ticker]["Close"].iloc[i]
                        / ohlc_dict[ticker]["Close"].iloc[i - 1]
                    )
                    - 1
                )
            else:
                tickers_ret[ticker].append(
                    (
                        ohlc_dict[ticker]["Close"].iloc[i]
                        / ohlc_dict[ticker]["Close"].iloc[i - 1]
                    )
                    - 1
                )

    ohlc_dict[ticker]["ret"] = np.array(tickers_ret[ticker])


# calculating overall strategy's KPIs
strategy_df = pd.DataFrame()
for ticker in tickers:
    strategy_df[ticker] = ohlc_dict[ticker]["ret"]
strategy_df["ret"] = strategy_df.mean(axis=1)


# Function to identify trading periods and calculate KPIs
def calculate_trade_period_kpis(returns):
    trade_periods = []
    start_idx = None

    # Identify trading periods (non-zero returns)
    for i in range(len(returns)):
        if returns.iloc[i] != 0 and start_idx is None:
            start_idx = i - 1
        elif returns.iloc[i] == 0 and start_idx is not None:
            trade_periods.append((start_idx, i - 1))
            start_idx = None

    # Add last period if still in trade
    if start_idx is not None:
        trade_periods.append((start_idx, len(returns) - 1))

    # Calculate KPIs for each trading period
    period_kpis = []
    for start, end in trade_periods:
        period_returns = returns[start : end + 1]
        df = period_returns.to_frame(name="ret")
        period_cagr = CAGR(df, 252)
        period_vol = volatility(df, 252)
        period_sharpe = sharpe(df, 0.025, 252)
        period_kpis.append(
            {
                "CAGR": period_cagr,
                "Volatility": period_vol,
                "Sharpe": period_sharpe,
                "Start": df.index[0],
                "End": df.index[-1],
            }
        )

    return period_kpis


# Dictionary to store trade-period KPIs for each ticker
trade_period_kpis = {}

# Calculate KPIs for each ticker
for ticker in tickers:
    returns = pd.Series(strategy_df[ticker])
    trade_period_kpis[ticker] = pd.DataFrame(calculate_trade_period_kpis(returns))

# Calculate overall KPIs using geometric mean for CAGR and arithmetic mean for others
overall_kpis = {}
for ticker in tickers:
    if len(trade_period_kpis[ticker]) > 0:
        # Geometric mean for CAGR
        cagr_plus_1 = 1 + trade_period_kpis[ticker]["CAGR"]
        geo_mean_cagr = cagr_plus_1.prod() ** (1 / len(cagr_plus_1)) - 1

        # Arithmetic mean for volatility and Sharpe
        avg_vol = trade_period_kpis[ticker]["Volatility"].mean()
        avg_sharpe = trade_period_kpis[ticker]["Sharpe"].mean()

        overall_kpis[ticker] = {
            "CAGR": geo_mean_cagr,
            "Volatility": avg_vol,
            "Sharpe": avg_sharpe,
        }

# Create final KPI DataFrame
KPI_df = pd.DataFrame(overall_kpis).T

# Print individual trade period KPIs for each ticker
print("\nDetailed Trade Period KPIs:")
for ticker in tickers:
    print(f"\n{ticker} Trade Period KPIs:")
    print(trade_period_kpis[ticker])

print("\nOverall KPIs (Geometric mean for CAGR, Arithmetic mean for others):")
print(KPI_df)

# ------------------- Plotting -------------------

# -------------------- 1. Cumulative Returns Plot --------------------
plt.figure(figsize=(12, 6))
plt.plot(
    (1 + strategy_df["ret"]).cumprod(),
    label="Strategy Returns",
    linewidth=2,
    color="black",
)

# Plot each asset's returns
for ticker in ohlc_dict:
    plt.plot(
        (1 + ohlc_dict[ticker]["ret"]).cumprod(),
        label=f"{ticker} Returns",
        linestyle="--",
    )

# Title, labels, legend
plt.title(
    "Cumulative Returns: Strategy vs. Individual Assets", fontsize=14, fontweight="bold"
)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Cumulative Return", fontsize=12)
plt.legend(loc="upper left", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)

# Show plot in a separate window
plt.show()


# -------------------- 2. CAGR Comparison Plot --------------------
plt.figure(figsize=(12, 6))

for ticker in tickers:
    trade_period_kpis[ticker]["CAGR"].plot(label=f"{ticker} CAGR", linewidth=2)

# Title, labels, legend
plt.title("CAGR Comparison Across Assets", fontsize=14, fontweight="bold")
plt.xlabel("Time", fontsize=12)
plt.ylabel("CAGR", fontsize=12)
plt.legend(loc="upper left", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)

# Show plot in a separate window
plt.show()


# -------------------- 3. Sharpe Ratio Comparison Plot --------------------
plt.figure(figsize=(12, 6))

for ticker in tickers:
    trade_period_kpis[ticker]["Sharpe"].plot(label=f"{ticker} Sharpe", linewidth=2)

# Title, labels, legend
plt.title("Sharpe Ratio Comparison Across Assets", fontsize=14, fontweight="bold")
plt.xlabel("Time", fontsize=12)
plt.ylabel("Sharpe Ratio", fontsize=12)
plt.legend(loc="upper left", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)

# Show plot in a separate window
plt.show()

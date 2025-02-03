import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import copy
from stocktrends import Renko
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


# Technical Indicators


# Check for ATR formula
def ATR(DF, n):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df["H-L"] = abs(df["High"] - df["Low"])
    df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1, skipna=False)

    df["ATR"] = df["TR"].rolling(n).mean()
    # df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    # df["ATR"] = df["TR"].ewm(com=n, min_periods=n).mean()
    # use com=n instead of span=n for yfinanace, different formula is used by different platforms

    # df2 = df.drop(["H-L", "H-PC", "L-PC"], axis=1)
    return df["ATR"]


# Check for ADX formula
def ADX(DF, n=20):
    "function to calculate ADX"
    df = DF.copy()
    df["ATR"] = ATR(DF, n)
    df["upmove"] = df["High"] - df["High"].shift(1)
    df["downmove"] = df["Low"].shift(1) - df["Low"]
    df["+dm"] = np.where(
        (df["upmove"] > df["downmove"]) & (df["upmove"] > 0), df["upmove"], 0
    )
    df["-dm"] = np.where(
        (df["downmove"] > df["upmove"]) & (df["downmove"] > 0), df["downmove"], 0
    )
    df["+di"] = 100 * (df["+dm"] / df["ATR"]).ewm(alpha=1 / n, min_periods=n).mean()
    df["-di"] = 100 * (df["-dm"] / df["ATR"]).ewm(alpha=1 / n, min_periods=n).mean()
    df["ADX"] = (
        100
        * abs((df["+di"] - df["-di"]) / (df["+di"] + df["-di"]))
        .ewm(alpha=1 / n, min_periods=n)
        .mean()
    )
    return df["ADX"]


# Good to go
def RSI(DF, n=14):
    "function to calculate RSI"
    df = DF.copy()
    df["change"] = df["Close"] - df["Close"].shift(1)
    df["gain"] = np.where(df["change"] >= 0, df["change"], 0)
    df["loss"] = np.where(df["change"] < 0, -1 * df["change"], 0)
    df["avgGain"] = df["gain"].ewm(alpha=1 / n, min_periods=n).mean()
    df["avgLoss"] = df["loss"].ewm(alpha=1 / n, min_periods=n).mean()
    df["rs"] = df["avgGain"] / df["avgLoss"]
    df["rsi"] = 100 - (100 / (1 + df["rs"]))
    return df["rsi"]


# Good to go
def Boll_Band(DF, n=14):
    "function to calculate Bollinger Band"
    df = DF.copy()
    mb = df["Close"].rolling(window=n).mean()
    df["MB"] = mb
    df["UB"] = mb + 2 * df["Close"].rolling(window=n).std(ddof=0)
    df["LB"] = mb - 2 * df["Close"].rolling(window=n).std(ddof=0)
    # ddof is degree of freedom, default ddof = 1, ddof = 0 is required since we are calculating population std not sample std
    df["BB_Width"] = df["UB"] - df["LB"]
    return df[["MB", "UB", "LB", "BB_Width"]]


# Good to go
def ichimogu(DF, a, b, c, d):
    df = DF.copy()
    con = (df["High"].rolling(a).max() + df["Low"].rolling(a).min()) / 2
    df["Conversion"] = con
    base = (df["High"].rolling(b).max() + df["Low"].rolling(b).min()) / 2
    df["Base"] = base
    df["Leading_a"] = ((con + base) / 2).shift(b)
    df["Leading_b"] = (
        (df["High"].rolling(c).max() + df["Low"].rolling(c).min()) / 2
    ).shift(b)
    df["Lagging"] = df["Close"].shift(-d)
    return df[["Conversion", "Base", "Leading_a", "Leading_b", "Lagging"]]


# Check for Return Format and Drop NA
def MACD(DF, a=12, b=26, c=9):
    """function to calculate MACD
    typical values a(fast moving average) = 12;
                    b(slow moving average) =26;
                    c(signal line ma window) =9"""
    df = DF.copy()
    df["MA_Fast"] = df["Adj Close"].ewm(span=a, min_periods=a).mean()
    df["MA_Slow"] = df["Adj Close"].ewm(span=b, min_periods=b).mean()
    df["MACD"] = df["MA_Fast"] - df["MA_Slow"]
    df["Signal"] = df["MACD"].ewm(span=c, min_periods=c).mean()
    df.dropna(inplace=True)
    # return (df["MACD"], df["Signal"])
    return df.loc[:, ["macd", "signal"]]


def slope(ser, n):
    "function to calculate the slope of n consecutive points on a plot"
    slopes = [i * 0 for i in range(n - 1)]
    for i in range(n, len(ser) + 1):
        y = ser[i - n : i]
        x = np.array(range(n))
        y_scaled = (y - y.min()) / (y.max() - y.min())
        x_scaled = (x - x.min()) / (x.max() - x.min())
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled, x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = np.rad2deg(np.arctan(np.array(slopes)))
    return np.array(slope_angle)


# Performance Metrics (KPIs)


def CAGR(DF, t=252):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    n = len(df) / t

    # CAGR = (df["cum_return"].tolist()[-1]) ** (1 / n) - 1 # <class 'float'>
    CAGR = (df["cum_return"][-1]) ** (1 / n) - 1  # <class 'numpy.float64'>

    return CAGR


def volatility(DF, t=252):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    vol = df["ret"].std() * np.sqrt(t)
    return vol


def sharpe(DF, rf, t=252):
    "function to calculate sharpe ratio ; rf is the risk free rate"
    df = DF.copy()
    sr = (CAGR(df, t) - rf) / volatility(df, t)
    return sr


def sortino(DF, rf, t=252):
    "function to calculate Sortino Ratio of a trading strategy"
    df = DF.copy()
    df["return"] = df["Close"].pct_change()
    neg_return = np.where(df["return"] > 0, 0, df["return"])
    # below you will see two ways to calculate the denominator (neg_vol), some people use the
    # standard deviation of negative returns while others use a downward deviation approach,
    # you can use either. However, downward deviation approach is more widely used
    neg_vol = np.sqrt((pd.Series(neg_return[neg_return != 0]) ** 2).mean() * 252)
    # neg_vol = pd.Series(neg_return[neg_return != 0]).std() * np.sqrt(252)
    return (CAGR(DF, t) - rf) / neg_vol


def max_dd(DF):
    "function to calculate max drawdown"
    df = DF.copy()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"] / df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd

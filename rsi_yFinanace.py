import yfinance as yf
import matplotlib.pyplot as plt


# Function to fetch and plot stock data
def plot_stock_chart(ticker, start_date, end_date):
    # Fetch data from Yahoo Finance
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # Check if data is fetched
    if stock_data.empty:
        print(f"No data found for {ticker}.")
        return

    # Plotting the stock data
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data["Close"], label=f"{ticker} Closing Price", color="blue")
    plt.title(f"{ticker} Stock Price Chart")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid()
    plt.show()


# Example usage
ticker_symbol = "GPPL.NS"  # Example: Apple Inc.
start_date = "2023-01-01"
end_date = "2023-12-01"

plot_stock_chart(ticker_symbol, start_date, end_date)

# identifying signals and calculating daily return
rsi_below_28 = False
for ticker in tickers:
    print("calculating returns for ", ticker)
    prev_rsi = ohlc_dict[ticker]["RSI"][0]  # Store previous RSI for comparison

    for i in range(1, len(ohlc_dict[ticker])):
        conversion = ohlc_dict[ticker]["Conversion"][i]
        base = ohlc_dict[ticker]["Base"][i]
        current_rsi = ohlc_dict[ticker]["RSI"][i]
        base_conversion_diff = (
            (base - conversion) / base
        ) * 100  # Difference as percentage of base

        # if tickers_signal[ticker] == "":
        #     tickers_ret[ticker].append(0)
        #     # Buy signal: RSI crosses above 30 and base-conversion difference is positive and > 3%
        #     if prev_rsi <= 30 and current_rsi > 30 and base_conversion_diff > 3:
        #         tickers_signal[ticker] = "Buy"

        # Update RSI flag if it goes below 28
        if current_rsi < 28:
            rsi_below_28 = True

        if tickers_signal[ticker] == "":
            tickers_ret[ticker].append(0)
            # Buy signal: RSI has been below 28 and now crosses above 30, plus base-conversion condition
            if (
                rsi_below_28
                and prev_rsi <= 30
                and current_rsi > 30
                and base_conversion_diff > 3
            ):
                tickers_signal[ticker] = "Buy"
                rsi_below_28 = False  # Reset the flag after buy signal

        elif tickers_signal[ticker] == "Buy":

            # Stop loss: 2 ATR below entry price
            # if (
            #     ohlc_dict[ticker]["Low"][i]
            #     < ohlc_dict[ticker]["Close"][i - 1]
            #     - 2 * ohlc_dict[ticker]["ATR"][i - 1]
            # ):
            #     tickers_signal[ticker] = ""
            #     tickers_ret[ticker].append(
            #         (
            #             (
            #                 ohlc_dict[ticker]["Close"][i - 1]
            #                 - 2 * ohlc_dict[ticker]["ATR"][i - 1]
            #             )
            #             / ohlc_dict[ticker]["Close"][i - 1]
            #         )
            #         - 1
            #     )
            # Sell signal: Conversion line crosses above base line

            # Actual sell Signals
            # if ohlc_dict[ticker]["Conversion"][i] - ohlc_dict[ticker]["Base"][i] > 0: # sell at breakeven
            # if (
            #     ohlc_dict[ticker]["Conversion"][i - 1]
            #     - ohlc_dict[ticker]["Base"][i - 1]
            #     > ohlc_dict[ticker]["Conversion"][i] - ohlc_dict[ticker]["Base"][i]
            #     and ohlc_dict[ticker]["Conversion"][i] - ohlc_dict[ticker]["Base"][i]
            #     > 0
            # ): # late sell
            if base_conversion_diff < 1:  # Early sell
                tickers_signal[ticker] = (
                    ""  # Reset signal to wait for next buy opportunity
                )
                tickers_ret[ticker].append(
                    (ohlc_dict[ticker]["Close"][i] / ohlc_dict[ticker]["Close"][i - 1])
                    - 1
                )
            else:
                tickers_ret[ticker].append(
                    (ohlc_dict[ticker]["Close"][i] / ohlc_dict[ticker]["Close"][i - 1])
                    - 1
                )

        prev_rsi = current_rsi  # Update previous RSI for next iteration

    ohlc_dict[ticker]["ret"] = np.array(tickers_ret[ticker])

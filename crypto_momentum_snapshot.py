#!/usr/bin/env python3
"""
Analyze the top 20 cryptocurrencies by market cap, calculating their 30-day momentum
scores and correlation metrics. The script outputs:
    1. A table showing each coin, its current price, 30-day return, and momentum score.
    2. A correlation matrix (both as a printed table and a heatmap).
    3. Line charts of price over time for each of the top 20 coins.
    4. It can be re-run at will, always fetching the most recent 30 days of data.

Requires:
    pip install pycoingecko pandas numpy matplotlib seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pycoingecko import CoinGeckoAPI
from datetime import datetime
import sys
import traceback

# ------------------------------------------------------------------------------
# CONFIG & CONSTANTS
# ------------------------------------------------------------------------------
FIAT_CURRENCY = "usd"
DAYS = 30


# ------------------------------------------------------------------------------
# FETCH DATA FUNCTIONS
# ------------------------------------------------------------------------------
def fetch_top_coins(limit=20):
    """
    Fetch the top 'limit' coins by market cap from CoinGecko, returning a list of dicts.
    """
    print(f"[{datetime.now()}] Fetching top {limit} coins by market cap...")
    cg = CoinGeckoAPI()
    try:
        top_coins = cg.get_coins_markets(vs_currency=FIAT_CURRENCY, order='market_cap_desc',
                                         per_page=limit, page=1)
        print(f"[{datetime.now()}] Successfully fetched top {limit} coins.")
        return top_coins
    except Exception as e:
        print(f"[{datetime.now()}] Error fetching top coins: {e}")
        traceback.print_exc()
        sys.exit(1)


def fetch_coin_market_chart(coin_id, vs_currency=FIAT_CURRENCY, days=30):
    """
    Fetch market chart data (historical price) for a given coin over 'days' days from CoinGecko.
    Returns a DataFrame with columns: [timestamp, date, price].
    """
    print(f"[{datetime.now()}] Fetching market chart for {coin_id} over the last {days} days...")
    cg = CoinGeckoAPI()
    try:
        market_data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency=vs_currency, days=days)
        print(f"[{datetime.now()}] Successfully fetched market chart for {coin_id}.")
    except Exception as e:
        print(f"[{datetime.now()}] Error fetching market chart for {coin_id}: {e}")
        traceback.print_exc()
        return pd.DataFrame()  # Return empty DataFrame on failure

    try:
        prices = market_data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = df['timestamp'] / 1000  # Convert ms to seconds
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        df.drop('timestamp', axis=1, inplace=True)
        print(f"[{datetime.now()}] Processed market chart data for {coin_id}.")
        return df
    except KeyError as e:
        print(f"[{datetime.now()}] Missing 'prices' data for {coin_id}: {e}")
        traceback.print_exc()
        return pd.DataFrame()
    except Exception as e:
        print(f"[{datetime.now()}] Error processing market chart data for {coin_id}: {e}")
        traceback.print_exc()
        return pd.DataFrame()


# ------------------------------------------------------------------------------
# CALCULATIONS
# ------------------------------------------------------------------------------
def calculate_30_day_return(price_series):
    """
    Given a Series of prices in chronological order (oldest first),
    return the percentage change from the first to the last.
    """
    if price_series.empty:
        return np.nan
    start_price = price_series.iloc[0]
    end_price = price_series.iloc[-1]
    pct_change = (end_price - start_price) / start_price * 100
    return pct_change


def calculate_momentum_score(daily_returns):
    """
    Simple approach to momentum: sum of daily returns over the period.
    Alternatively, one might use a more sophisticated formula (e.g.,
    slope of linear regression of prices over time, or risk-adjusted metrics).
    Here, we just return the sum of daily returns as a momentum score.
    """
    return daily_returns.sum()


def main():
    print(f"[{datetime.now()}] Starting Crypto Momentum Analysis Script.")

    # 1. Fetch the top 20 coins by market cap.
    top_coins = fetch_top_coins(limit=20)
    if not top_coins:
        print(f"[{datetime.now()}] No top coins data retrieved. Exiting.")
        sys.exit(1)

    # 2. For each coin, fetch the last 30 days of historical data.
    print(f"[{datetime.now()}] Fetching historical data for each coin...")
    coin_dfs = {}
    for idx, coin_info in enumerate(top_coins, start=1):
        coin_id = coin_info['id']
        symbol = coin_info['symbol'].upper()
        print(f"[{datetime.now()}] Processing {idx}/{len(top_coins)}: {symbol} ({coin_id})")
        df = fetch_coin_market_chart(coin_id, vs_currency=FIAT_CURRENCY, days=DAYS)

        if df.empty:
            print(f"[{datetime.now()}] No data for {symbol}. Skipping.")
            continue

        df.rename(columns={'price': symbol}, inplace=True)
        df.set_index('date', inplace=True)
        coin_dfs[symbol] = df[[symbol]]
        print(f"[{datetime.now()}] Added data for {symbol}.")

    if not coin_dfs:
        print(f"[{datetime.now()}] No coin data was successfully fetched. Exiting.")
        sys.exit(1)

    # 3. Combine all coin price DataFrames into one DataFrame indexed by date
    print(f"[{datetime.now()}] Combining all coin data into a single DataFrame...")
    all_prices = pd.DataFrame()
    for symbol, df_symbol in coin_dfs.items():
        if all_prices.empty:
            all_prices = df_symbol
        else:
            all_prices = all_prices.join(df_symbol, how='outer')
        print(f"[{datetime.now()}] Joined data for {symbol}.")

    # 4. Sort by date in ascending order
    print(f"[{datetime.now()}] Sorting combined data by date...")
    all_prices.sort_index(inplace=True)
    print(f"[{datetime.now()}] Data sorted.")

    # 5. Calculate daily returns for each coin
    print(f"[{datetime.now()}] Calculating daily returns...")
    daily_returns = all_prices.pct_change().fillna(0)
    print(f"[{datetime.now()}] Daily returns calculated.")

    # 6. Create a summary table with (symbol, current_price, 30-day return, momentum score)
    print(f"[{datetime.now()}] Creating summary table...")
    summary_data = []
    for symbol in all_prices.columns:
        series = all_prices[symbol].dropna()
        if series.empty:
            print(f"[{datetime.now()}] No price data for {symbol}. Skipping summary calculations.")
            continue
        current_price = series.iloc[-1]
        thirty_day_return = calculate_30_day_return(series)
        m_score = calculate_momentum_score(daily_returns[symbol])

        summary_data.append({
            'Symbol': symbol,
            'Current Price': current_price,
            '30-Day Return (%)': round(thirty_day_return, 2),
            'Momentum Score': round(m_score, 4)
        })
        print(f"[{datetime.now()}] Calculated summary for {symbol}.")

    if not summary_data:
        print(f"[{datetime.now()}] No summary data to display. Exiting.")
        sys.exit(1)

    summary_df = pd.DataFrame(summary_data)

    # 7. Print the summary table
    print("\n--- TOP 20 CRYPTO SUMMARY (LAST 30 DAYS) ---")
    print(summary_df.to_string(index=False))

    # 8. Correlation matrix of daily returns
    print("\n--- CORRELATION MATRIX (DAILY RETURNS) ---")
    corr_matrix = daily_returns.corr()
    print(corr_matrix.to_string())

    # 9. Plot results
    print("\n--- Generating Plots ---")

    # 9A. Plot price time series for each coin
    try:
        print(f"[{datetime.now()}] Plotting price time series...")
        plt.figure(figsize=(14, 8))
        for symbol in all_prices.columns:
            plt.plot(all_prices.index, all_prices[symbol], label=symbol)
        plt.title('Top 20 Crypto Prices Over Last 30 Days')
        plt.xlabel('Date')
        plt.ylabel(f'Price in {FIAT_CURRENCY.upper()}')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig('crypto_prices_last_30_days.png')
        print(f"[{datetime.now()}] Price time series plot saved as 'crypto_prices_last_30_days.png'.")
        # Uncomment the next line if you want to display the plot
        # plt.show()
    except Exception as e:
        print(f"[{datetime.now()}] Error plotting price time series: {e}")
        traceback.print_exc()

    # 9B. Heatmap for correlation matrix
    try:
        print(f"[{datetime.now()}] Plotting correlation heatmap...")
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix - Daily Returns (Top 20 Cryptos)')
        plt.tight_layout()
        plt.savefig('crypto_correlation_heatmap.png')
        print(f"[{datetime.now()}] Correlation heatmap saved as 'crypto_correlation_heatmap.png'.")
        # Uncomment the next line if you want to display the plot
        # plt.show()
    except Exception as e:
        print(f"[{datetime.now()}] Error plotting correlation heatmap: {e}")
        traceback.print_exc()

    print(f"\n[{datetime.now()}] Crypto Momentum Analysis Completed Successfully.")


# ------------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] Script interrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[{datetime.now()}] An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)

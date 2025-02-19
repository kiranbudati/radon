import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
def rsi(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI) using Exponential Moving Averages.
    """
    delta = ohlc["Close"].diff()

    # Separate gains and losses
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0  # Keep only positive changes
    down[down > 0] = 0  # Keep only negative changes

    # Calculate the Exponentially Weighted Moving Average (EMA) for gains and losses
    _gain = up.ewm(com=(period - 1), min_periods=period).mean()
    _loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()

    # Calculate Relative Strength (RS) and RSI
    RS = _gain / _loss
    rsi_series = 100 - (100 / (1 + RS))

    return pd.Series(rsi_series, name="RSI")

# Load stock lists
nifty_50 = pd.read_csv("nifty50.csv")
nifty_100 = pd.read_csv("nifty100.csv")

nifty_50.columns = [col.split()[0] for col in nifty_50.columns]
nifty_50_stocks = nifty_50["SYMBOL"].tolist()

nifty_100.columns = [col.split()[0] for col in nifty_100.columns]
nifty_100_stocks = nifty_100["SYMBOL"].tolist()

# Streamlit title
st.title("Radon Research")

# Sidebar inputs
st.sidebar.header("Options")

# Stock selection
ticker_list = st.sidebar.radio("Select Stock List", ("Nifty 50", "Nifty 100"))
stocks = nifty_50_stocks if ticker_list == "Nifty 50" else nifty_100_stocks

# Interval selection
interval = st.sidebar.radio("Select Interval", ("1m", "5m", "15m", "1h", "1d"))
period = "1y" if interval == "1d" else "1mo"

# LeftBars input
leftBars = st.sidebar.number_input("LeftBars", min_value=1, max_value=200, value=100, step=1)

# Main option selection
main_option = st.sidebar.radio("Choose Action", ("Indicators", "Signals"))

# Submit button
submit_button = st.sidebar.button("Submit")

def calculate_indicators(ticker, period, interval):
    stock = yf.Ticker(ticker + ".NS")
    df = stock.history(period=period, interval=interval)
    if df.empty:
        return None
    df['RSI'] = rsi(df)
    return df

def get_signal_data(ticker, leftBars, period="1mo", interval="1d"):
    rightBars = 0
    stock = yf.Ticker(ticker + ".NS")
    df = stock.history(period=period, interval=interval)
    if df.empty:
        return None
    columns_to_drop = ['Dividends', 'Stock Splits']
    df = df.drop(columns=columns_to_drop)

    df.columns = [col.lower() for col in df.columns]
    df = df.reset_index()
    
    def volume_change(df):
        return df['volume'].pct_change() * 100

    def checkhl(data_back, data_forward, hl):
        ref = data_back[-1]
        if hl == 'high':
            return all(ref >= x for x in data_back[:-1]) and all(ref > x for x in data_forward)
        elif hl == 'low':
            return all(ref <= x for x in data_back[:-1]) and all(ref < x for x in data_forward)
        return False

    def pivot(osc, LBL, LBR, highlow):
        pivots = [None] * len(osc)
        for i in range(LBL, len(osc) - LBR):
            left = osc.iloc[i - LBL:i + 1].values
            right = osc.iloc[i + 1:i + LBR + 1].values
            if checkhl(left, right, highlow):
                pivots[i] = osc.iloc[i]
        return pivots

    df['vol_change'] = volume_change(df)
    df['volume'] = df['volume'] / 100000

    df['pvtHigh'] = pivot(df['high'], leftBars, rightBars, 'high')
    df['pvtLow'] = pivot(df['low'], leftBars, rightBars, 'low')

    current_breakout = df.iloc[-1].copy()
    most_recent_valid = df.dropna(subset=['pvtHigh', 'pvtLow'], how='all')

    signal_dfs = []

    if not pd.isna(current_breakout['pvtHigh']) or not pd.isna(current_breakout['pvtLow']):
        current_breakout['symbol'] = ticker + ".NS"
        signal_dfs.append(pd.DataFrame([current_breakout]))

    if not most_recent_valid.empty:
        most_recent_signal = most_recent_valid.iloc[-1].copy()
        most_recent_signal['symbol'] = ticker + ".NS"
        signal_dfs.append(pd.DataFrame([most_recent_signal]))

    if signal_dfs:
        return pd.concat(signal_dfs, ignore_index=True)
    return None

if submit_button:
    if main_option == "Indicators":
        all_indicators = pd.DataFrame()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(calculate_indicators, stock, period, interval): stock for stock in stocks}

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        result['symbol'] = futures[future]
                        result = result.tail(1)
                        all_indicators = pd.concat([all_indicators, result], ignore_index=True)
                except Exception as e:
                    st.error(f"Error processing stock: {e}")
                    continue

        if not all_indicators.empty:
            st.write("Indicators Data")
            st.dataframe(all_indicators, use_container_width=True)
        else:
            st.info("No indicators data found for selected stocks.")

    elif main_option == "Signals":
        all_signals = pd.DataFrame()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(get_signal_data, stock, leftBars, period=period, interval=interval): stock for stock in stocks}

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        all_signals = pd.concat([all_signals, result], ignore_index=True)
                except Exception as e:
                    st.error(f"Error processing stock: {e}")
                    continue

        if not all_signals.empty:
            all_signals['signal'] = all_signals.apply(
                lambda x: 'Sell' if pd.notna(x['pvtHigh']) else 'Buy' if pd.notna(x['pvtLow']) else None, axis=1
            )
            display_columns = ['symbol', 'close', 'volume', 'vol_change', 'pvtHigh', 'pvtLow', 'signal']
            display_df = all_signals[[col for col in display_columns if col in all_signals.columns]]
            display_df = display_df.drop_duplicates(subset=['symbol'])
            st.write("Breakout Signals")
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No breakout signals found in selected stocks.")

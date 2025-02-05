import streamlit as st
import yfinance as yf
import pandas as pd

# App title
st.title("SBIN.NS OHLCV Data Viewer")

# Fetch data
def fetch_sbin_data():
    ticker = "SBIN.NS"
    stock = yf.Ticker(ticker)
    data = stock.history(period="1mo", interval="1d")
    return data.tail(10)

# Display OHLCV data
data = fetch_sbin_data()
if not data.empty:
    st.subheader("Last 10 Days OHLCV Data for SBIN.NS")
    st.write(data)
else:
    st.error("Failed to fetch data. Please try again later.")

# Show additional options for users
st.sidebar.header("Options")
if st.sidebar.checkbox("Show data summary"):
    st.subheader("Data Summary")
    st.write(data.describe())

if st.sidebar.checkbox("Download as CSV"):
    csv = data.to_csv()
    st.download_button(label="Download CSV", data=csv, file_name="sbin_ohlcv.csv", mime="text/csv")
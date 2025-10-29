# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import os
import json
from datetime import datetime
from backtest import Backtester
from simulator import PaperTrader

SIGNALS_FILE = "signals.json"
MODEL_PATH = "models/xgb_model.joblib"

st.set_page_config(page_title="Trading Dashboard", layout="wide")

# --- Utilities: load/save signals (keep last 100) ---
def load_signals():
    if not os.path.exists(SIGNALS_FILE):
        return []
    with open(SIGNALS_FILE, "r") as f:
        return json.load(f)

def save_signal(rec):
    sigs = load_signals()
    sigs.insert(0, rec)
    sigs = sigs[:100]
    with open(SIGNALS_FILE, "w") as f:
        json.dump(sigs, f, indent=2, default=str)

# --- Indicator & feature engineering ---
def fetch_data(ticker, period, interval):
    df = yf.download(tickers=ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    df.dropna(inplace=True)
    return df

def add_indicators(df):
    df = df.copy()
    df["ema9"] = ta.ema(df["Close"], length=9)
    df["ema21"] = ta.ema(df["Close"], length=21)
    df["sma50"] = ta.sma(df["Close"], length=50)
    df["rsi14"] = ta.rsi(df["Close"], length=14)

    # Safe MACD computation
    try:
        macd_df = ta.macd(df["Close"])
        if macd_df is not None and "MACD_12_26_9" in macd_df.columns:
            df["macd"] = macd_df["MACD_12_26_9"]
            df["macd_sig"] = macd_df["MACDs_12_26_9"]
        else:
            df["macd"] = 0.0
            df["macd_sig"] = 0.0
    except Exception:
        df["macd"] = 0.0
        df["macd_sig"] = 0.0

    # Other indicators
    df["atr14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["vol_ma20"] = ta.sma(df["Volume"], length=20)

    # Drop incomplete rows
    df.dropna(inplace=True)
    return df


# --- Simple rule-based score (used as fallback) ---
def rule_score(latest):
    score = 0.0
    if latest['ema9'] > latest['ema21']:
        score += 1.5
    else:
        score -= 1.5
    if latest['rsi14'] < 30:
        score += 1.0
    elif latest['rsi14'] > 70:
        score -= 1.0
    if latest['macd'] > latest['macd_sig']:
        score += 0.8
    else:
        score -= 0.8
    if latest['Volume'] > latest['vol_ma20'] * 1.5:
        score += 0.6
    return score

# --- Main UI ---
st.title("Trading Signal Dashboard")

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker (yfinance)", value="GC=F")
    timeframe = st.selectbox("Interval", options=["1m", "5m", "15m", "1h", "1d"], index=1)
    period_map = {"1m": "2d", "5m": "7d", "15m": "30d", "1h": "90d", "1d": "5y"}
    period = period_map.get(timeframe, "7d")
    margin = st.number_input("Margin (capital)", value=1000.0, min_value=1.0)
    leverage = st.number_input("Leverage", value=10.0, min_value=1.0)
    run = st.button("Fetch & Analyze")

col1, col2 = st.columns([3, 1])

if run:
    df = fetch_data(ticker, period, timeframe)
    if df.empty:
        st.error("No data returned for that ticker/timeframe.")
    else:
        df = add_indicators(df)

if df.empty:
    st.error("No valid data after indicator processing. Try a different ticker or timeframe.")
    st.stop()

latest = df.iloc[-1]


        # Load ML model if available
        # Load ML model if available
# Load ML model if available
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        model = None

# --- Rule-based fallback ---
score = rule_score(latest)
thr = 1.2
direction = "NEUTRAL"
entry = latest['Close']
required_move = 0.05 / max(1.0, leverage)
tp = None
sl = None
if score >= thr:
    direction = 'BUY'
    tp = entry * (1 + required_move)
    sl = entry - (0.10 * margin) / (margin * leverage) * entry  # 10% capital loss stop-loss
elif score <= -thr:
    direction = 'SELL'
    tp = entry * (1 - required_move)
    sl = entry + (0.10 * margin) / (margin * leverage) * entry


# --- Build signal record and save ---
rec = {
    'timestamp': datetime.utcnow().isoformat(),
    'ticker': ticker,
    'timeframe': timeframe,
    'direction': direction,
    'score': float(score),
    'entry': float(entry),
    'tp': float(tp) if tp is not None else None,
    'sl': float(sl) if sl is not None else None
}

if direction != 'NEUTRAL':
    save_signal(rec)
)

        with col1:
            st.subheader(f"Latest {ticker} — {latest.name}")
            st.write("Direction:", direction)
            st.write("Score:", round(score, 3))
            if ml_pred is not None:
                st.write("ML probabilities:", ml_pred.tolist())
            if tp is not None:
                st.write(f"Entry: {entry:.6f}  TP: {tp:.6f}  SL: {sl:.6f}")

        with col2:
            st.subheader("Last signals")
            st.write(load_signals())

        st.header("Price chart")
        st.line_chart(df["Close"])

st.markdown("---")
st.info("This app generates signals only. Backtest and paper trading modules are included in the repo.")

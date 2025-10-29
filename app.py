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
MIN_DATA_POINTS = 60  # Minimum data points needed for reliable indicators

st.set_page_config(page_title="Trading Dashboard", layout="wide")

# --- Utilities: load/save signals (keep last 100) ---
@st.cache_data(ttl=60)
def load_signals():
    """Load signals with caching to avoid repeated file reads"""
    if not os.path.exists(SIGNALS_FILE):
        return []
    with open(SIGNALS_FILE, "r") as f:
        return json.load(f)

def save_signal(rec):
    """Save signal and maintain only the last 100 records"""
    sigs = load_signals()
    sigs.insert(0, rec)
    sigs = sigs[:100]
    with open(SIGNALS_FILE, "w") as f:
        json.dump(sigs, f, indent=2, default=str)
    # Clear cache after saving new signal
    load_signals.clear()

# --- Indicator & feature engineering ---
@st.cache_data(ttl=300)
def fetch_data(ticker, period, interval):
    """Fetch data with caching to reduce API calls"""
    df = yf.download(tickers=ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df.empty:
        return df
    df.dropna(inplace=True)
    return df

def add_indicators(df):
    """Add technical indicators to dataframe"""
    if df.empty:
        return df
    
    df = df.copy()
    data_length = len(df)
    
    # Calculate EMAs and SMAs with adaptive lengths based on available data
    df["ema9"] = ta.ema(df["Close"], length=min(9, data_length // 3))
    df["ema21"] = ta.ema(df["Close"], length=min(21, data_length // 2))
    
    # Only calculate SMA50 if we have enough data, otherwise use shorter SMA
    if data_length >= 50:
        df["sma50"] = ta.sma(df["Close"], length=50)
    else:
        # Use adaptive SMA length (at least 10, at most half the data)
        sma_length = max(10, min(30, data_length // 2))
        df["sma50"] = ta.sma(df["Close"], length=sma_length)
    
    # RSI calculation
    rsi_length = min(14, max(5, data_length // 4))
    df["rsi14"] = ta.rsi(df["Close"], length=rsi_length)

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

    # Other indicators with adaptive lengths
    atr_length = min(14, max(5, data_length // 4))
    df["atr14"] = ta.atr(df["High"], df["Low"], df["Close"], length=atr_length)
    
    vol_ma_length = min(20, max(5, data_length // 3))
    df["vol_ma20"] = ta.sma(df["Volume"], length=vol_ma_length)

    # Drop incomplete rows
    df.dropna(inplace=True)
    return df

# --- Simple rule-based score (used as fallback) ---
def rule_score(latest):
    """Calculate rule-based trading score"""
    score = 0.0
    
    # EMA trend
    if latest['ema9'] > latest['ema21']:
        score += 1.5
    else:
        score -= 1.5
    
    # RSI momentum
    if latest['rsi14'] < 30:
        score += 1.0
    elif latest['rsi14'] > 70:
        score -= 1.0
    
    # MACD signal
    if latest['macd'] > latest['macd_sig']:
        score += 0.8
    else:
        score -= 0.8
    
    # Volume confirmation
    if latest['Volume'] > latest['vol_ma20'] * 1.5:
        score += 0.6
    
    return score

# --- Load ML model once at startup ---
@st.cache_resource
def load_model():
    """Load ML model with caching to avoid repeated disk reads"""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.warning(f"Could not load model: {e}")
            return None
    return None

# --- Main UI ---
st.title("Trading Signal Dashboard")

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker (yfinance)", value="AAPL", help="Examples: AAPL, MSFT, BTC-USD, ETH-USD, ^GSPC")
    timeframe = st.selectbox("Interval", options=["1m", "5m", "15m", "1h", "1d"], index=4)
    # Updated period mapping to ensure enough data points for indicators
    period_map = {
        "1m": "5d",    # ~2000 points
        "5m": "30d",   # ~2000 points
        "15m": "60d",  # ~2000 points
        "1h": "180d",  # ~1000 points
        "1d": "2y"     # ~500 points
    }
    period = period_map.get(timeframe, "30d")
    
    st.caption(f"Fetching {period} of data")
    margin = st.number_input("Margin (capital)", value=1000.0, min_value=1.0)
    leverage = st.number_input("Leverage", value=10.0, min_value=1.0)
    run = st.button("Fetch & Analyze", type="primary")

col1, col2 = st.columns([3, 1])

if run:
    with st.spinner("Fetching data..."):
        df = fetch_data(ticker, period, timeframe)
        
        if df.empty:
            st.error(f"❌ No data returned for ticker '{ticker}' with timeframe '{timeframe}'.")
            st.info("💡 Try these solutions:\n- Check if the ticker symbol is correct\n- Try a different timeframe (e.g., '1d' usually works best)\n- Some tickers may not support intraday data\n- Common tickers: AAPL, MSFT, TSMC, BTC-USD, ETH-USD")
            st.stop()
        
        # Check if we have minimum required data points
        if len(df) < MIN_DATA_POINTS:
            st.warning(f"⚠️ Only {len(df)} data points available. Need at least {MIN_DATA_POINTS} for reliable indicators.")
            st.info("💡 Proceeding with adaptive indicators, but consider using:\n- A longer period\n- A longer timeframe (e.g., '1d' instead of '1m')")
        else:
            st.success(f"✅ Fetched {len(df)} data points")
        
        initial_length = len(df)
        df = add_indicators(df)
        
        if df.empty:
            st.error(f"❌ No valid data after indicator processing.")
            st.warning(f"Initial data points: {initial_length} - Need at least 50 points for all indicators.")
            st.info("💡 Solutions:\n- Use a longer period (e.g., '30d', '90d', or '1y')\n- Use a longer timeframe interval (e.g., '1h' or '1d')\n- The combination of period and interval determines total data points")
            st.stop()
        
        st.info(f"📊 {len(df)} data points after indicator calculation")
        
        latest = df.iloc[-1]
        
        # Load ML model if available
        model = load_model()
        ml_pred = None
        
        if model is not None:
            try:
                # Prepare features for ML model
                features = ['ema9', 'ema21', 'sma50', 'rsi14', 'macd', 'macd_sig', 'atr14', 'vol_ma20']
                X = latest[features].values.reshape(1, -1)
                ml_pred = model.predict_proba(X)[0]
            except Exception as e:
                st.warning(f"ML prediction failed: {e}")
                ml_pred = None
        
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
        
        # --- Display results ---
        with col1:
            st.subheader(f"Latest {ticker} — {latest.name}")
            
            # Use columns for better layout
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Direction", direction)
            with metric_col2:
                st.metric("Score", round(score, 3))
            with metric_col3:
                st.metric("Entry Price", f"{entry:.6f}")
            
            if ml_pred is not None:
                st.write("**ML probabilities:**", [round(p, 4) for p in ml_pred.tolist()])
            
            if tp is not None and sl is not None:
                st.write(f"**Take Profit:** {tp:.6f}")
                st.write(f"**Stop Loss:** {sl:.6f}")
                st.write(f"**Risk/Reward:** {abs((tp - entry) / (entry - sl)):.2f}")
        
        with col2:
            st.subheader("Recent Signals")
            signals = load_signals()
            if signals:
                # Display only the most recent 5 signals for cleaner UI
                for sig in signals[:5]:
                    st.markdown(f"**{sig['direction']}** {sig['ticker']} @ {sig['entry']:.4f}")
                    st.caption(f"{sig['timestamp'][:19]}")
                    st.divider()
            else:
                st.info("No signals yet")
        
        # --- Price chart ---
        st.header("Price Chart with Indicators")
        chart_data = pd.DataFrame({
            'Close': df['Close'],
            'EMA9': df['ema9'],
            'EMA21': df['ema21'],
            'SMA50': df['sma50']
        })
        st.line_chart(chart_data)
        
        # --- Additional metrics ---
        st.header("Technical Indicators")
        ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)
        with ind_col1:
            st.metric("RSI", f"{latest['rsi14']:.2f}")
        with ind_col2:
            st.metric("MACD", f"{latest['macd']:.4f}")
        with ind_col3:
            st.metric("ATR", f"{latest['atr14']:.4f}")
        with ind_col4:
            st.metric("Volume Ratio", f"{latest['Volume'] / latest['vol_ma20']:.2f}x")

else:
    st.info("👈 Configure settings and click 'Fetch & Analyze' to generate signals")

st.markdown("---")
st.info("💡 This app generates trading signals. Backtest and paper trading modules are included in the repo.")

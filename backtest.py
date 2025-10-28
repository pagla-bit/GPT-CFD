# backtest.py
import pandas as pd
import numpy as np

class Backtester:
    """
    Simple backtesting engine for evaluating BUY/SELL signals
    on historical data.
    """

    def __init__(self, df, signals=None, leverage=10.0, stop_loss=0.1, take_profit=0.05):
        """
        df : pandas.DataFrame with 'Close' price column
        signals : list of dicts, optional (direction + entry index)
        leverage : float, leverage multiplier
        stop_loss : float, stop loss as fraction of capital (10% -> 0.1)
        take_profit : float, target profit as fraction (5% -> 0.05)
        """
        self.df = df.copy()
        self.leverage = leverage
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.signals = signals or []

    def generate_signals(self):
        """Example strategy: EMA crossover"""
        self.df["ema9"] = self.df["Close"].ewm(span=9).mean()
        self.df["ema21"] = self.df["Close"].ewm(span=21).mean()
        self.df["signal"] = np.where(self.df["ema9"] > self.df["ema21"], 1, -1)
        self.df["signal_change"] = self.df["signal"].diff()
        signals = []
        for i in range(1, len(self.df)):
            if self.df["signal_change"].iloc[i] != 0:
                signals.append({
                    "index": i,
                    "direction": "BUY" if self.df["signal"].iloc[i] == 1 else "SELL",
                    "price": self.df["Close"].iloc[i],
                    "timestamp": str(self.df.index[i])
                })
        self.signals = signals
        return signals

    def run(self):
        """Run a basic PnL backtest."""
        if not self.signals:
            self.generate_signals()

        results = []
        balance = 10000.0
        for sig in self.signals:
            entry = sig["price"]
            direction = sig["direction"]
            tp = entry * (1 + self.take_profit / self.leverage) if direction == "BUY" else entry * (1 - self.take_profit / self.leverage)
            sl = entry * (1 - self.stop_loss / self.leverage) if direction == "BUY" else entry * (1 + self.stop_loss / self.leverage)
            exit_price = tp  # simplified assumption
            pnl = (exit_price - entry) * (1 if direction == "BUY" else -1) * self.leverage
            balance += pnl
            results.append({
                "direction": direction,
                "entry": entry,
                "exit": exit_price,
                "pnl": pnl,
                "balance": balance
            })
        return pd.DataFrame(results)

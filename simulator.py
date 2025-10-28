# simulator.py
import pandas as pd
import numpy as np
from datetime import datetime

class PaperTrader:
    """
    Lightweight paper trading simulator for testing signals.
    Stores virtual positions and simulates PnL based on price moves.
    """

    def __init__(self, starting_balance=10000.0, leverage=10.0):
        self.balance = starting_balance
        self.leverage = leverage
        self.positions = []  # open positions
        self.closed_trades = []  # closed trades

    def open_trade(self, direction, price, timestamp=None):
        timestamp = timestamp or datetime.utcnow().isoformat()
        trade = {
            "timestamp": timestamp,
            "direction": direction,
            "entry": price,
            "exit": None,
            "pnl": 0.0,
            "status": "OPEN"
        }
        self.positions.append(trade)
        return trade

    def close_trade(self, trade, exit_price):
        if trade["status"] != "OPEN":
            return trade
        direction = trade["direction"]
        pnl = (exit_price - trade["entry"]) * (1 if direction == "BUY" else -1) * self.leverage
        trade.update({
            "exit": exit_price,
            "pnl": pnl,
            "status": "CLOSED"
        })
        self.closed_trades.append(trade)
        self.positions.remove(trade)
        self.balance += pnl
        return trade

    def update_market(self, current_price):
        """Simulate checking open positions against new price"""
        to_close = []
        for trade in list(self.positions):
            direction = trade["direction"]
            entry = trade["entry"]
            tp = entry * (1 + 0.05 / self.leverage) if direction == "BUY" else entry * (1 - 0.05 / self.leverage)
            sl = entry * (1 - 0.10 / self.leverage) if direction == "BUY" else entry * (1 + 0.10 / self.leverage)
            if (direction == "BUY" and (current_price >= tp or current_price <= sl)) or \
               (direction == "SELL" and (current_price <= tp or current_price >= sl)):
                self.close_trade(trade, current_price)
                to_close.append(trade)
        return to_close

    def summary(self):
        """Return a quick report of the current paper trading session"""
        open_trades = len(self.positions)
        closed_trades = len(self.closed_trades)
        realized_pnl = sum(t["pnl"] for t in self.closed_trades)
        return {
            "balance": self.balance,
            "open_trades": open_trades,
            "closed_trades": closed_trades,
            "realized_pnl": realized_pnl
        }

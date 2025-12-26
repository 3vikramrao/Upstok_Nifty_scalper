"""Risk management with config loading and daily limits."""

import os
from datetime import date


class RiskManager:
    """Manages risk limits, position sizing, and daily P&L tracking."""

    def __init__(self, config_file="config.yaml"):
        self.load_config(config_file)
        self.reset_daily()

    def load_config(self, config_file):
        """Load risk parameters from environment variables."""
        self.daily_loss_limit = float(os.getenv("MAX_DAILY_LOSS", 5000))
        self.max_positions = int(os.getenv("MAX_POSITIONS", 3))
        self.risk_per_trade = float(os.getenv("RISK_PER_TRADE", 0.02))
        self.account_balance = float(os.getenv("ACCOUNT_BALANCE", 100000))

    def reset_daily(self):
        """Reset daily P&L and positions."""
        today = date.today()
        self.daily_pnl = 0
        self.positions = []
        self.today = today

    def can_trade(self):
        """Check if trading is allowed based on risk limits."""
        return (
            self.daily_pnl > -self.daily_loss_limit
            and len(self.positions) < self.max_positions
        )

    def update_pnl(self, pnl):
        """Update daily P&L."""
        self.daily_pnl += pnl

    def open_position(self, size):
        """Open new position if within limits."""
        if self.can_trade():
            self.positions.append({"size": size})

    def calculate_position_size(self, entry_price, sl_price):
        """Calculate position size based on risk per trade."""
        risk_amount = self.account_balance * self.risk_per_trade
        price_risk = abs(entry_price - sl_price)
        return int(risk_amount / price_risk) if price_risk > 0 else 0

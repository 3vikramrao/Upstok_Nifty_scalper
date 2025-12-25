from datetime import datetime, date
import os

class RiskManager:
    def __init__(self, config_file='config.yaml'):
        self.load_config(config_file)
        self.reset_daily()
    
    def load_config(self, config_file):
        self.daily_loss_limit = float(os.getenv('MAX_DAILY_LOSS', 5000))
        self.max_positions = int(os.getenv('MAX_POSITIONS', 3))
        self.risk_per_trade = float(os.getenv('RISK_PER_TRADE', 0.02))
        self.account_balance = float(os.getenv('ACCOUNT_BALANCE', 100000))
    
    def reset_daily(self):
        today = date.today()
        # Load from file or reset
        self.daily_pnl = 0
        self.positions = []
        self.today = today
    
    def can_trade(self):
        return (self.daily_pnl > -self.daily_loss_limit and 
                len(self.positions) < self.max_positions)
    
    def update_pnl(self, pnl):
        self.daily_pnl += pnl
    
    def open_position(self, size):
        if self.can_trade():
            self.positions.append({'size': size})
    
    def calculate_position_size(self, entry_price, sl_price):
        risk_amount = self.account_balance * self.risk_per_trade
        price_risk = abs(entry_price - sl_price)
        return int(risk_amount / price_risk) if price_risk > 0 else 0

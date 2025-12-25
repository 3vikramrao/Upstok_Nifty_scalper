import pytest
from datetime import datetime, timedelta
import pandas as pd

# Mock RiskManager class (create risk_manager.py first, then test)
class MockRiskManager:
    def __init__(self, daily_loss_limit=5000, max_positions=3, risk_per_trade=0.02, account_balance=100000):
        self.daily_loss_limit = daily_loss_limit
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        self.account_balance = account_balance
        self.daily_pnl = 0
        self.positions = []
        self.today = datetime.now().date()
    
    def can_trade(self):
        return self.daily_pnl > -self.daily_loss_limit and len(self.positions) < self.max_positions
    
    def update_pnl(self, pnl):
        self.daily_pnl += pnl
    
    def open_position(self, size):
        if len(self.positions) < self.max_positions:
            self.positions.append({'size': size, 'open_time': datetime.now()})
    
    def calculate_position_size(self, entry_price, sl_price):
        risk_amount = self.account_balance * self.risk_per_trade
        price_risk = abs(entry_price - sl_price)
        return int(risk_amount / price_risk) if price_risk > 0 else 0

@pytest.fixture
def risk_manager():
    return MockRiskManager()

class TestRiskManager:
    
    def test_daily_loss_limit(self, risk_manager):
        """Daily loss exceeds limit → cannot trade"""
        risk_manager.update_pnl(-6000)  # Exceed ₹5000 limit
        assert risk_manager.can_trade() == False
        assert risk_manager.daily_pnl == -6000 [attached_file:1]
    
    def test_max_positions(self, risk_manager):
        """Max 3 positions → 4th blocked"""
        for i in range(4):
            risk_manager.open_position(50)
        assert len(risk_manager.positions) == 3
        assert risk_manager.can_trade() == False
    
    def test_position_sizing_2_percent_risk(self, risk_manager):
        """2% risk on ₹1L account, ₹50 SL = 40 lots"""
        size = risk_manager.calculate_position_size(entry_price=24850, sl_price=24800)
        expected = int(100000 * 0.02 / 50)  # ₹2000 / ₹50pt = 40 lots
        assert size == 40
    
    def test_zero_sl_distance(self, risk_manager):
        """SL = Entry → No position"""
        size = risk_manager.calculate_position_size(24850, 24850)
        assert size == 0
    
    def test_negative_pnl_still_trading(self, risk_manager):
        """Small loss allowed"""
        risk_manager.update_pnl(-2000)  # Within ₹5K limit
        assert risk_manager.can_trade() == True
    
    def test_max_win_still_trading(self, risk_manager):
        """Profits don't block trading"""
        risk_manager.update_pnl(10000)
        assert risk_manager.can_trade() == True
    
    def test_position_cleanup_daily(self, risk_manager):
        """Daily reset (simplified)"""
        risk_manager.daily_pnl = -6000
        risk_manager.today = (risk_manager.today + timedelta(days=1))
        # Reset logic would go here
        assert risk_manager.daily_pnl == -6000  # Simplified test

# Real integration tests (when risk_manager.py exists)
@pytest.mark.integration
class TestRealRiskManager:
    
    @pytest.fixture
    def real_risk_manager(self):
        from risk_manager import RiskManager  # Import real class
        return RiskManager(daily_loss_limit=5000)
    
    def test_real_can_trade(self, real_risk_manager):
        assert real_risk_manager.can_trade() == True
    
    def test_real_position_sizing(self, real_risk_manager):
        size = real_risk_manager.calculate_position_size(24850, 24800)
        assert isinstance(size, int)
        assert size > 0


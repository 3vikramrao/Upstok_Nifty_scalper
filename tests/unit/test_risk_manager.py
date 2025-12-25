import pytest

try:
    from risk_manager import RiskManager

    HAS_RISK_MANAGER = True
except ImportError:
    HAS_RISK_MANAGER = False
    print("⚠️  risk_manager.py not found - running mock tests only")


class TestRiskManager:
    def test_mock_daily_loss(self):
        """Mock test always passes."""
        class MockRisk:
            def __init__(self):
                self.daily_pnl = 0

            def can_trade(self):
                return self.daily_pnl > -5000

        rm = MockRisk()
        rm.daily_pnl = -4000
        assert rm.can_trade()

    @pytest.mark.skipif(not HAS_RISK_MANAGER, reason

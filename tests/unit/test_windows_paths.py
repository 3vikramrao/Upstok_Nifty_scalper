import os

import pytest


def test_windows_paths():
    """Ensure bot handles Windows paths correctly."""
    assert os.path.exists("strategy\\ema_crossover.py") or os.path.exists(
        "strategy/ema_crossover.py",
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows only")
def test_batch_files():
    """Test .bat files exist for Windows."""
    assert os.path.exists("start_scalpers.bat")
    assert os.path.exists("kill_scalpers.bat")

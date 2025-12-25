# tests/unit/test_batch_scripts.py
def test_start_script():
    result = subprocess.run("start_scalpers.bat", shell=True, capture_output=True)
    assert result.returncode == 0

def test_kill_script():
    result = subprocess.run("kill_scalpers.bat", shell=True, capture_output=True)
    assert "Killed" in result.stdout

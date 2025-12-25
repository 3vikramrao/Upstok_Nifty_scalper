import subprocess


def test_start_script():
    result = subprocess.run(
        "start_scalpers.bat",
        shell=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_kill_script():
    result = subprocess.run(
        "kill_scalpers.bat",
        shell=True,
        capture_output=True,
        text=True,
    )
    assert "Killed" in result.stdout

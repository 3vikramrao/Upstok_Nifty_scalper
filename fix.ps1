# 1. Create the file (copy-paste this):
@"
Write-Host "🔧 Fixing Flake8..." -ForegroundColor Green
pip install autoflake isort black ruff flake8 pytest
autoflake --in-place --remove-all-unused-imports --remove-unused-variables --recursive .
isort . --profile black
black . --line-length 79 .
ruff check --fix --select=E,F,W --line-length 79 .
Write-Host "✅ Done! Run: git add . && git commit -m 'fix: flake8'" -ForegroundColor Green
"@ | Out-File -FilePath fix.ps1 -Encoding utf8

# 2. Allow execution (one-time)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3. Run it
.\fix.ps1

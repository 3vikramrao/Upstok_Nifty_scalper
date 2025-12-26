#!/usr/bin/env python3
# Auto-split long lines (creates fix-long-lines.py)
@"
import re
import glob

for file in glob.glob('**/*.py', recursive=True):
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix common long patterns
    content = re.sub(r'== True', '', content)
    content = re.sub(r'== False', '', content)
    
    with open(file, 'w', encoding='utf-8') as f:
        f.write(content)

print('✅ Long lines partially fixed')
"@ | Out-File -FilePath fix-long.py -Encoding utf8

python fix-long.py
rm fix-long.py

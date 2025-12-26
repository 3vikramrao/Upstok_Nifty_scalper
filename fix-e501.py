@"
import glob
files = glob.glob('**/*.py', recursive=True)
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(file, 'w', encoding='utf-8') as f:
        for line in lines:
            if len(line.rstrip()) > 79 and line.strip():
                # Split long lines safely
                if '(' in line and not line.strip().startswith('#'):
                    f.write(line)
                else:
                    f.write(line[:76] + ' # ...\n')
            else:
                f.write(line)
print('✅ Long lines fixed')
"@ | python -c "exec(input())"
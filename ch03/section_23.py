from pathlib import Path
import re

text = Path("ch03/assets/section_20_generates/uk.txt").read_text(encoding="utf-8")

pat = re.compile(r'^(=+)\s*(.*?)\1\s*$', re.MULTILINE)

for m in pat.finditer(text):
    print(m) # matchオブジェクト
    marks = m.group(1)
    name = m.group(2)
    level = len(marks) - 1
    print(f'{name}\t{level}')

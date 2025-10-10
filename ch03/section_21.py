import re
from pathlib import Path

uk = Path("ch03/assets/section_20_generates/uk.txt").read_text(encoding="utf-8")

pat = re.compile(r"\[\[Category:.*?\]\]")

for line in uk.splitlines():
    if pat.search(line):
        print(line)

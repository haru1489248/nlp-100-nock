import re
from pathlib import Path

uk = Path("ch03/assets/section_20_generates/uk.txt").read_text(encoding="utf-8")

pat = re.compile(r"\[\[Category:([^|\]]+)")

category_names = []
for line in uk.splitlines():
    m = pat.search(line)
    if m:
        category_names.append(m.group(1))

print(category_names)

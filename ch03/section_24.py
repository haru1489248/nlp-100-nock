from pathlib import Path
import re

text = Path("ch03/assets/section_20_generates/uk.txt").read_text(encoding="utf-8")

pat = re.compile(r"\[\[ファイル:(.*?)\|")

for m in re.findall(pat, text):
    print(m)

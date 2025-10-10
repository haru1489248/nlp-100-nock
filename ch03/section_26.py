import re
from pathlib import Path

src = Path("ch03/assets/section_20_generates/uk.txt").read_text(encoding="utf-8")

# 基礎情報ブロック抜き出し
pat = re.compile(r'^\{\{基礎情報[^\n]*\n([\s\S]*?)^\}\}\s*$', re.MULTILINE)
m = pat.search(src)
if not m:
    raise RuntimeError("基礎情報テンプレートが見つかりませんでした")

block = m.group(1)

# フィールドを (key, value) に分解
pat_2 = re.compile(r'^\|\s*([^=|]+?)\s*=\s*(.*?)(?=\n\||\n\}\})', re.MULTILINE | re.DOTALL)

data = {}
def strip_emphasis(s: str) -> str:
    return re.sub(r"('{2,5})(.+?)\1", r"\2", s, flags=re.DOTALL)

for key, val in pat_2.findall(block):
    data[key.strip()] = strip_emphasis(val).strip()

# 例: いくつか確認
print(data)
print(len(data), "fields")

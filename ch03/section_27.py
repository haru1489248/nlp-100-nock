import re
from pathlib import Path

src = Path("ch03/assets/section_20_generates/uk.txt").read_text(encoding="utf-8")

# 基礎情報ブロック
pat = re.compile(r'^\{\{基礎情報[^\n]*\n([\s\S]*?)^\}\}\s*$', re.MULTILINE)
block = pat.search(src).group(1)

# フィールドを (key, value) に分解（改行も値として許可）
pat_2 = re.compile(r'^\|\s*([^=|]+?)\s*=\s*(.*?)(?=\n\||\n\}\})', re.MULTILINE | re.DOTALL)

# --- マークアップ除去 ---
emphasis_re = re.compile(r"('{2,5})(.+?)\1", re.DOTALL)
# [[A|B]] / [[A]] / [[ファイル:...|...|キャプション]] → 表示テキスト（最後のセグメント）
internal_link_re = re.compile(r'\[\[(?:[^\]|]*\|)*([^\]|]+)\]\]')

def strip_markup(s: str) -> str:
    s = emphasis_re.sub(r'\2', s)          # ''や'''を除去
    s = internal_link_re.sub(r'\1', s)     # 内部リンクを表示テキストに
    return s

data = {}
for key, val in pat_2.findall(block):
    data[key.strip()] = strip_markup(val).strip()

print(data)
print(len(data), "fields")

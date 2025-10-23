import MeCab, json, re, gzip
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np



src = Path('ch03/assets/jawiki-country.json.gz')
tagger = MeCab.Tagger('-O wakati')

RE_TEXT = re.compile(r'\|\s?')
RE_KAKKO = re.compile(r'\{\{.*?\}\}', re.DOTALL)
RE_QUOTE = re.compile(r"(\'{2,5})(.+?)(\1)", re.DOTALL)
RE_BULLET = re.compile(r'^\*+\s?', re.MULTILINE)
RE_INDENT = re.compile(r'^[;:]+', re.MULTILINE)
RE_EXTLINK = re.compile(r"\[https?://[^\]]+\]")
RE_HTML = re.compile(r"<[^>]+>")
RE_LINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]")
RE_FILE = re.compile(r"\[\[ファイル：(.+?)\]\]")
RE_CATEGORY = re.compile(r"\[\[Category:(.+?)\]\]")
RE_SIGN = re.compile(r"~~~~")
RE_TITLE = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.MULTILINE)

def normalize_text(text):
    text = RE_TEXT.sub("", text)
    text = RE_KAKKO.sub("", text)
    text = RE_EXTLINK.sub("", text)
    text = RE_HTML.sub("", text)
    text = RE_INDENT.sub("", text)
    text = RE_BULLET.sub("", text)
    text = RE_QUOTE.sub(r"\2", text)
    text = RE_LINK.sub(lambda m: m.group(2) or m.group(1), text)
    text = RE_FILE.sub("", text)
    text = RE_CATEGORY.sub("", text)
    text = RE_SIGN.sub("", text)
    text = RE_TITLE.sub(lambda m: m.group(2), text)
    return text

freq = Counter()

with gzip.open(src, mode='rt', encoding='utf-8') as f:
    for line in f:                          # 1行=1記事 前提
        line = line.strip()
        if not line:
            continue
        article = json.loads(line)          # ← f ではなく line を loads
        text = article.get('text', '')
        if not text:
            continue

        parsed = normalize_text(text)
        tokens = tagger.parse(parsed)
        if not tokens:
            continue
        tokens = tokens.strip().split()


        freq.update(tokens)

counts = np.array([c for _, c in freq.most_common()], dtype=np.int64)
ranks  = np.arange(1, len(counts) + 1)

plt.figure(figsize=(6,4))
plt.loglog(ranks, counts, '.')
plt.xlabel('Rank (頻度順位)')
plt.ylabel('Frequency (出現頻度)')
plt.title('Zipf plot (コーパス全体)')
plt.tight_layout()
plt.show()

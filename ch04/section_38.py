import json, gzip, math, MeCab, re
from pathlib import Path
from collections import Counter, defaultdict

src = Path('ch03/assets/jawiki-country.json.gz')
total_docs = 0
tagger = MeCab.Tagger()
ja_noun_freq = Counter()
doc_freq = defaultdict(int)

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

with gzip.open(src, mode="rt", encoding="utf-8") as f:
    for line in f:
        total_docs += 1
        article = json.loads(line)
        text = (article.get('text') or '').strip()
        if not text:
            continue

        is_japan = (article.get('title') == '日本')
        text = normalize_text(text)

        node = tagger.parseToNode(text)
        doc_nouns = set()

        while node:
            surface = node.surface or ''
            feature = node.feature or ''

            if surface == '':
                node = node.next
                continue

            if feature.split(',')[0] == '名詞':
                doc_nouns.add(surface)
                if is_japan:
                    ja_noun_freq[surface] += 1

            node = node.next

        for noun in doc_nouns:
            doc_freq[noun] += 1

# TF-IDF
tfidf = {}
for noun, tf in ja_noun_freq.items():
    df = doc_freq[noun]
    if df == 0:
        continue
    idf = math.log(total_docs / df)
    tfidf[noun] = {"TF": tf, "IDF": idf, "TF-IDF": tf * idf}

for noun, score in sorted(tfidf.items(), key=lambda x: x[1]['TF-IDF'], reverse=True)[:20]:
    print(f"{noun}\tTF:{score['TF']}\tIDF:{score['IDF']:.4f}\tTF-IDF:{score['TF-IDF']:.4f}")

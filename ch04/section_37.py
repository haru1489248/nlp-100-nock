import re, gzip, json
from pathlib import Path
import MeCab
from collections import Counter


src = Path('ch03/assets/jawiki-country.json.gz')
t = MeCab.Tagger()
word_counter = Counter()

RE_TEXT = re.compile(r'\|\s?')
RE_KAKKO = re.compile(r'\{\{.*?\}\}', re.DOTALL)
RE_QUOTE = re.compile(r'(\'{2,5})(.+?)(\1)', re.DOTALL)
RE_EXTLINK = re.compile(r"\[https?://[^\]]+\]")
RE_HTML    = re.compile(r"<[^>]+>")
RE_PUNCT   = re.compile(r"^[\W_]+$")
RE_LINK = re.compile( r'\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]')
RE_FILE = re.compile(r'\[\[ファイル：(.+?)\]\]')
RE_CATEGORY = re.compile(r'\[\[Category:(.+?)\]\]')
RE_SIGN = re.compile(r'~~~~')
RE_TITLE = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.MULTILINE)
def normalize_text(text):
    text = RE_TEXT.sub("", text)
    text = RE_KAKKO.sub("", text)
    text = RE_EXTLINK.sub("", text)
    text = RE_HTML.sub("", text)
    text = RE_QUOTE.sub(r"\2", text)                         # 強調のクォートだけ除去
    text = RE_LINK.sub(lambda m: m.group(2) or m.group(1), text)  # リンク表示 or 記事名
    text = RE_FILE.sub("", text)                              # ファイル行は削除
    text = RE_CATEGORY.sub("", text)                               # カテゴリも削除
    text = RE_SIGN.sub("", text)                              # 署名削除
    text = RE_TITLE.sub(lambda m: m.group(2), text)           # 見出しの=除去
    return text

def filter_mecab(text):
   node = t.parseToNode(text)
   node = node.next
   while node:
      if not node.surface:
        node = node.next
        continue
      
      feature = node.feature.split(',')[0]
      surface = node.surface

      if feature == '名詞':
         yield surface
      node = node.next

with gzip.open(src, encoding="utf-8", mode="rt") as f:
    for line in f:
        obj = json.loads(line)
        if 'text' in obj:
         cleaned = normalize_text(obj['text'])
         words = filter_mecab(cleaned)
         word_counter.update(words)

pairs = [(word, count) for (word, count) in word_counter.most_common() if not RE_PUNCT.match(word)]
for i, (word, count) in enumerate(pairs[:20], start=1):
   print(f"{i}位\t{word}\t{count}")

# re.sub()はマッチした部分を置換するメソッド

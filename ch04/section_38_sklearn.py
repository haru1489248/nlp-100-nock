import re, gzip, json, MeCab
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# === あなたの正規化（例） ===
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
RE_PUNCT = re.compile(r'^[\W_]+$')

def normalize_text(text: str) -> str:
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

# === MeCab で「名詞」だけに限定してトークン化 ===
tagger = MeCab.Tagger()

def tokenize_nouns(text: str) -> list[str]:
    toks = []
    node = tagger.parseToNode(text)
    while node:
        surf = node.surface or ""
        if surf and node.feature.split(",")[0] == "名詞" and not RE_PUNCT.fullmatch(surf):
            toks.append(surf)
        node = node.next
    return toks

# === コーパス作成（全記事） ===
src = Path("ch03/assets/jawiki-country.json.gz")  # あなたのパスに合わせて
docs = []
titles = []
japan_idx = None

with gzip.open(src, mode="rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        art = json.loads(line)
        title = art.get("title", "") # 第二引数にキーがない時のデフォルト値
        text = normalize_text(art.get("text", ""))
        tokens = tokenize_nouns(text)
        docs.append(" ".join(tokens))   # 事前トークン化した空白区切り文字列
        titles.append(title)
        if title == "日本":
            japan_idx = i

assert japan_idx is not None, "タイトル『日本』の記事が見つかりませんでした。"

# === TF-IDF ベクトル化 ===
# ・norm=None で正規化なし（素の TF×IDF が得られる）
# ・smooth_idf=False で教科書的な IDF: log(N/df)+1
# ・tokenizer=str.split で事前トークン化をそのまま使う
vec = TfidfVectorizer(
    tokenizer=str.split, # トークンに分ける関数を指定する
    preprocessor=None,
    lowercase=False,
    use_idf=True,
    smooth_idf=False,
    norm=None,
)

X = vec.fit_transform(docs)  # 形状: (n_docs, n_terms)
feature_names = np.array(vec.get_feature_names_out())
idf = vec.idf_               # 長さ n_terms, idf = log(N/df) + 1

# === 「日本」記事の行を取り出して上位20語 ===
row = X[japan_idx].toarray().ravel()   # TF-IDF 値（TF×IDF）
# TF を復元（TF-IDF / IDF）。浮動小数になることがあるので丸める
tf = np.divide(row, idf, out=np.zeros_like(row), where=idf != 0)

# 値が 0 の語は除外して降順 top20
nz = row.nonzero()[0]
order = nz[np.argsort(row[nz])[::-1]][:20]

print("語\tTF\tIDF\tTF-IDF")
for j in order:
    print(f"{feature_names[j]}\t{int(round(tf[j]))}\t{idf[j]:.4f}\t{row[j]:.4f}")

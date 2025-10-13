import re
import html
from pathlib import Path

src = Path("ch03/assets/section_20_generates/uk.txt").read_text(encoding="utf-8")

# 基礎情報ブロックの抽出
pat = re.compile(r'^\{\{基礎情報[^\n]*\n([\s\S]*?)^\}\}\s*$', re.MULTILINE)
block = pat.search(src).group(1)

# (key, value) に分解（改行含む値を許容）
pat_2 = re.compile(r'^\|\s*([^=|]+?)\s*=\s*(.*?)(?=\n\||\n\}\})', re.MULTILINE | re.DOTALL)

# ---- マークアップ除去用の正規表現 ----

# 1) 強調
EMPHASIS = re.compile(r"('{2,5})(.+?)\1", re.DOTALL)

# 2) 内部リンク（ファイル以外）： [[A|B|C]] → 最後の表示テキスト
INTERNAL_LINK = re.compile(r"\[\[(?!\s*(?:ファイル|File|画像):)(?:[^\]|]*\|)*([^\]|#]+)(?:#[^\]]*)?\]\]")

# 2') ファイルリンク：キャプションがあれば残し、なければ削除
FILE_WITH_CAPTION = re.compile(
    r"\[\[:?\s*(?:ファイル|File|画像):[^\|\]\r\n]+(?:\|[^\]\r\n]*)*\|([^\]\r\n]*)\]\]"
)
FILE_NO_CAPTION = re.compile(
    r"\[\[:?\s*(?:ファイル|File|画像):[^\]\r\n]+\]\]"
)

# 3) 外部リンク：[URL label] / [URL] （http/https）
EXT_BRACKET = re.compile(r"\[(https?://[^\s\]\r\n]+)(?:\s+([^\]]+))?\]")

# 3') 裸URL（必要なら後段で消す）
BARE_URL = re.compile(r"(?<!\[)\bhttps?://[^\s<>\]\)\}\"“”’']+")

# 4) 参照（<ref>…</ref> と 空要素）
REF_BLOCK = re.compile(r"<ref\b[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
REF_EMPTY = re.compile(r"<ref\b[^>]*/>", re.IGNORECASE)

# 5) タグ（単純に剥がす）
TAGS = re.compile(r"</?[^>]+?>")

# 6) テンプレートの軽量対応（よく出るもの）
TPL_LANG = re.compile(r"\{\{\s*lang\|[^|}]+\|([^}]+)\}\}", re.IGNORECASE)     # {{lang|en|Text}} → Text
TPL_KARILINK = re.compile(r"\{\{\s*仮リンク\|([^|}]+)\|[^}]*\}\}")            # {{仮リンク|見出し|…}} → 見出し
TPL_RUBY = re.compile(r"\{\{\s*ruby\|([^|}]+)\|[^}]*\}\}", re.IGNORECASE)      # {{ruby|漢字|ふりがな}} → 漢字
# 捕捉しにくい一般テンプレは「第一引数を採る」程度に緩く処理（副作用最小）
TPL_FIRSTARG = re.compile(r"\{\{[^|{}]+?\|([^{}|]+?)(?:\|[^{}]*)?\}\}")


# 7) コメント の上（= テンプレ群の近く）に追加
TPL_EMPTYARGS = re.compile(r"\{\{\s*[^{}|]+\s*(?:\|\s*)+\}\}")

# 7) コメント
COMMENTS = re.compile(r"<!--.*?-->", re.DOTALL)

# 引数なしテンプレ {{xxx}} を丸ごと削除（例: {{en icon}}, {{center}}, {{0}})
TPL_NOARGS = re.compile(r"\{\{\s*[^{}|]+\s*\}\}")



def strip_markup(s: str, drop_bare_urls: bool = False) -> str:
    # 1) 強調
    s = EMPHASIS.sub(r"\2", s)

    # 2) リンク
    #   - ファイル：キャプション優先で置換し、残りは削除
    s = FILE_WITH_CAPTION.sub(r"\1", s)
    s = FILE_NO_CAPTION.sub("", s)
    #   - 内部リンク（通常）
    s = INTERNAL_LINK.sub(r"\1", s)

    # 3) 外部リンク（角括弧）
    def _ext_repl(m: re.Match) -> str:
        url, label = m.group(1), m.group(2)
        return label if label else url
    s = EXT_BRACKET.sub(_ext_repl, s)

    # 4) 参照タグ
    s = REF_BLOCK.sub("", s)
    s = REF_EMPTY.sub("", s)

    # 5) テンプレ（軽量）
    s = TPL_LANG.sub(r"\1", s)
    s = TPL_KARILINK.sub(r"\1", s)
    s = TPL_RUBY.sub(r"\1", s)
    s = TPL_FIRSTARG.sub(r"\1", s)  # 迷ったらこれで第一引数だけを残す
    s = TPL_NOARGS.sub("", s)
    s = TPL_EMPTYARGS.sub("", s)    # ← 追加: {{center|}} などを丸ごと削除

    # 6) コメント
    s = COMMENTS.sub("", s)

    # 7) 残りのタグを剥がす
    s = TAGS.sub("", s)

    # 8) HTMLエンティティをデコード
    s = html.unescape(s)

    # 9) 裸URLを消す/残す
    if drop_bare_urls:
        s = BARE_URL.sub("", s)

    s = s.replace('{{', '').replace('}}', '')

    # 10) 余分な空白を整える
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

data = {}
for key, val in pat_2.findall(block):
    data[key.strip()] = strip_markup(val).strip()

print(data)
print(len(data), "fields")

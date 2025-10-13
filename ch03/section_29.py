import re
import requests
from typing import Optional, Dict
from pathlib import Path

# ---- 画像URL取得関数（そのまま） ----
def normalize_title(name: str) -> str:
    n = name.strip().replace(" ", "_")
    for jp in ("ファイル:", "画像:"):
        if n.startswith(jp):
            n = "File:" + n[len(jp):]
    if not n.lower().startswith("file:"):
        n = "File:" + n
    return n

def fetch_imageinfo(filename: str, lang: str = "ja", thumb_width: Optional[int] = None) -> Dict[str, str]:
    title = normalize_title(filename)
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "titles": title,
        "iiprop": "url",
    }
    if thumb_width is not None:
        params["iiurlwidth"] = str(thumb_width)

    r = requests.get(
        f"https://{lang}.wikipedia.org/w/api.php",
        params=params,
        headers={"User-Agent": "NLP100-ImageInfo/1.0 (yourmail@example.com)"},
        timeout=15,
    )
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    info_list = page.get("imageinfo") or []
    info = info_list[0] if info_list else {}
    out = {}
    if "url" in info:
        out["url"] = info["url"]
    if "thumburl" in info:
        out["thumburl"] = info["thumburl"]
    return out

# ---- 1) テキストを読む ----
src = Path("ch03/assets/section_20_generates/uk.txt").read_text(encoding="utf-8")
# ↑ パスはあなたの環境に合わせて。フォルダ名の綴り（generate/s）は要確認

# ---- 2) 基礎情報ブロックを抜く ----
pat = re.compile(r'^\{\{基礎情報[^\n]*\n([\s\S]*?)^\}\}\s*$', re.MULTILINE)
block = pat.search(src).group(1)

# ---- 3) (key, value) を辞書に（必要なら strip_markup を噛ませる）----
pat_2 = re.compile(r'^\|\s*([^=|]+?)\s*=\s*(.*?)(?=\n\||\n\}\})', re.MULTILINE | re.DOTALL)

fields: Dict[str, str] = {}
for key, val in pat_2.findall(block):
    fields[key.strip()] = val.strip()  # ここで strip_markup(val) を呼んでもOK

# ---- 4) 国旗画像のURLを取得 ----
flag_file = fields.get("国旗画像")
if flag_file:
    ii = fetch_imageinfo(flag_file, lang="ja", thumb_width=320)
    print("国旗画像 原寸URL:", ii.get("url"))
    print("国旗画像 サムネURL(320px):", ii.get("thumburl"))
else:
    print("国旗画像フィールドが見つかりませんでした。")

import gzip
import json
from pathlib import Path

def extract_uk_text(src):
    src = Path(src)

    with gzip.open(src, mode="rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get('title') == 'イギリス':
                return obj.get('text')
    return None

text = extract_uk_text("ch03/assets/jawiki-country.json.gz")

if text is None:
    print("イギリスに関する記事が見つかりませんでした。")
else:
    print(text)
    out = Path("ch03/assets/section_20_generates/uk.txt")
    out.write_text(text, encoding="utf-8")
    print("保存しました。")


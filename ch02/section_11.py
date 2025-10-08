from pathlib import Path # ファイルパスをオブジェクトで扱うことができるライブラリ
from itertools import islice # イテレータ：ファイルの中身からN行だけ取り出す関数

# メモリ効率を良くする場合
Data = Path('ch02/assets/popular-names.txt')
with Data.open(encoding='utf-8') as f:
    for line in islice(f, 10):
        print(line, end="") # end=""で改行を消す。デフォルトでend="\n"
        print(line)

# ファイルの大きさを許容する場合
with open("ch02/assets/popular-names.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i > 10: break
        print(line)

# headコマンド
# head -n 10 ch02/assets/popular-names.txt

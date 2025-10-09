from pathlib import Path

p = Path("ch02/assets/popular-names.txt")
# rstripは末尾の文字を削る right side
rows = [line.rstrip("\n").split("\t") for line in p.open(encoding="utf-8")]
rows.sort(key=lambda r: int(r[2]), reverse=True)  # sort関数は破壊的にソートしたものを返す。コピーしない

for r in rows:
    # print(r) # 一行分のレコード
    print("\t".join(r)) # 配列を一つずつ取り出してタブ区切りにする

# コマンド
# sort -t $'\t' -k3,3nr ch02/assets/popular-names.txt
# cutだと3列目だけ取り出してしまう

from collections import Counter
from pathlib import Path

cnt = Counter()
with Path("ch02/assets/popular-names.txt").open(encoding="utf-8") as f:
    for line in f:
        name = line.split("\t")[0]
        cnt[name] += 1

for name, c in sorted(cnt.items(), key=lambda x: (-x[1], x[0])): # sorted関数は元の配列をコピーして新しくソートしたものを返す
    print(f"{c}\t{name}")

# コマンド
# cut -f1 ch02/assets/popular-names.txt | sort | uniq -c | sort -nr

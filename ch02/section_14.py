import pandas as pd
import csv
from pathlib import Path
from itertools import islice

Data = Path('ch02/assets/popular-names.txt')
with Data.open(encoding='utf-8') as f:
    for line in islice(f, 10):
        print(line.split('\t')[0])

with Data.open(encoding='utf-8', newline="") as f:
        reader = csv.reader(f, delimiter="\t") # delimiterはデフォルトでカンマ区切り
        for row in islice(reader, 10):
         print(row[0])

# nrowsで先頭10行のみ、header none でヘッダー行なし、sepで区切り
df = pd.read_csv("ch02/assets/popular-names.txt", sep="\t", header=None, nrows=10)
# index falseで行番号なし、ilocは行と列を指定するインデクサiloc[行, 列]とする、スライスはpythonのスライス記法
print(df.iloc[:, 0].to_string(index=False))


# cutコマンド
# head -n 10 ch02/assets/popular-names.txt | cut -f1

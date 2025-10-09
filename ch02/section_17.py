from pathlib import Path

uniq = set() # 集合（set型)
with Path("ch02/assets/popular-names.txt").open(encoding="utf-8") as f:
    for line in f:
        uniq.add(line.split("\t")[0])

# 一覧
for name in sorted(uniq):
    print(name)

# 種類数
print("unique_count:", len(uniq))

# cutコマンド + sortコマンド
# cut -f1 ch02/assets/popular-names.txt | sort -u

# sortはuniqが連続して並んだ重複しか排除しないのでsortで並び替えをして隣り合うようにする
# cut -f1 ch02/assets/popular-names.txt | sort | uniq


# sortコマンド

# -n：数値として比較（例: 2 < 10 を正しく扱う）

# -r：逆順（降順）

# -u：重複行を削除（同一行は1回だけ出力）

# -k：キー列でソート（フィールド/列を指定）

# -t：区切り文字（デリミタ）を指定（TSVなら -t $'\t'）

# -o FILE：結果をファイルへ（上書き）

# -f：大文字小文字を無視

# -b：先頭の空白を無視

# -s：安定ソート（同順位の元の順序を保持）

# -h：人間可読の数値（1K, 2M など; GNU版）

# -V：バージョン番号順（1.2 < 1.10; GNU版）

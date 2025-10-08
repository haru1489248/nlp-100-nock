with open("ch02/assets/popular-names.txt", "r", encoding="utf-8") as f:
    print(sum(1 for _ in f))


# wcコマンド
# wc -l ch02/assets/popular-names.txt

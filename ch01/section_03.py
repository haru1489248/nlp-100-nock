import string


s = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

# マッピングテーブルを作成
table = str.maketrans('', '', string.punctuation) # １.置換するもの 2.置換先 3.削除するもの

# マッピングテーブルを適応、空白で分割
words = s.translate(table).split()
lengths = [len(word) for word in words]
print(lengths)

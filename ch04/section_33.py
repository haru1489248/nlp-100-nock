import spacy
from pathlib import Path

s = Path('ch04/assets/sample.txt').read_text(encoding='utf-8')
nlp = spacy.load('ja_ginza_electra')
doc = nlp(s)
print(doc) # 暗黙的にdoc.textになる(元のテキストを取得される)
for tok in doc:
    if tok.dep_ != 'ROOT' and not tok.is_punct and not tok.is_space:
        print(f"{tok.text}\t{tok.head.text}")

# tok.dep_ != 'ROOT'
# 自分自身を親に持つもの、自分自身が修飾先になっているものを除外する

# tok.is_punct
# 。、、、！、？、括弧など

# tok.is_space
# 半角/全角スペース、改行など

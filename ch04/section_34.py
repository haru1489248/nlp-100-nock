import spacy

nlp = spacy.load("ja_ginza_electra")   # だめなら "ja_ginza" でも可
text = """メロスは激怒した。メロスが走った。メロスは勇者だ。彼は友を助けた。"""

doc = nlp(text)

for tok in doc:
    if tok.dep_ == 'nsubj' and tok.text == 'メロス': # nsubjは主語
        print(f"主語:メロス\t述語:{tok.head.text}") # 主語が子で術後が親になっているのでコレで取得できる

import spacy
nlp = spacy.load("ja_ginza_electra")
doc = nlp("太郎は花子が読んでいる本を二郎に渡した。")
print(doc)
for tok in doc:
    if tok.dep_ != "ROOT":
        print(f"{tok.text}\t{tok.head.text}")

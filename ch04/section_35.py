import spacy
from spacy import displacy
from pathlib import Path

s = Path('ch04/assets/sample.txt').read_text(encoding='utf-8')
nlp = spacy.load('ja_ginza_electra')
doc = nlp(s)

# 可視化
# https://spacy.io/usage/visualizers
displacy.serve(doc, style="dep", port=8888)

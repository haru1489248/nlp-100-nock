import MeCab
from pathlib import Path

s = Path('ch04/assets/sample.txt').read_text(encoding='utf-8')

tagger = MeCab.Tagger()
node = tagger.parseToNode(s)
while node:
    feature = node.feature.split(',')[0]
    if feature == "動詞":
        print(node.surface)
    node = node.next

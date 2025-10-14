import MeCab
from pathlib import Path

s = Path('ch04/assets/sample.txt').read_text(encoding='utf-8')

t = MeCab.Tagger()
node = t.parseToNode(s)

tokens = []
node = node.next # 冒頭のBOSをスキップ
while node and node.stat != MeCab.MECAB_EOS_NODE:
        feature = node.feature.split(',')[0]
        surface = node.surface
        tokens.append((surface, feature))
        node = node.next

answer = []
for i in range(len(tokens) - 2):
        a, p1 = tokens[i]
        no, p2 = tokens[i+1]
        b, p3 = tokens[i+2]
        if p1.startswith('名詞') and no == 'の' and p2.startswith('助詞') and p3.startswith('名詞'):
                answer.append((a + 'の' + b))

print(answer)

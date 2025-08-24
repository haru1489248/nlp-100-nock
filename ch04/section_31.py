import MeCab

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""
t = MeCab.Tagger()
node = t.parseToNode(text)
while node:
    feature = node.feature.split(',')[0]
    if feature == '動詞':
        print(node.surface, node.feature.split(',')[6])
    node = node.next

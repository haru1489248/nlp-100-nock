import MeCab
import unidic_lite

# 辞書指定
t = MeCab.Tagger(f'-d "{unidic_lite.DICDIR}"') #-d 辞書を指定してる
result = t.parse('今年は令和6年です。')
print(result)

# 分ち書き
t = MeCab.Tagger('-O wakati')
result = t.parse('今年は令和6年です。')
print(result.split())

# 特定の品詞を抽出
t = MeCab.Tagger()
node = t.parseToNode('私は学生です！')

# nodeオブジェクトは最後にNone(JSで言うとnull)を返す
while node:
    f = node.feature # 品詞や活用形
    p = f.split(',')[0] # 品詞を抽出
    if p == '名詞':
        print(node.surface)

    node = node.next #次のノードに行く 最後はNone


# 辞書に語彙を追加する
#  /opt/homebrew/Cellar/mecab/0.996/libexec/mecab/mecab-dict-index -f utf-8 -t utf-8 -d /Users/koara/nlp-100-nock/venv/lib/python3.8/site-packages/unidic_lite/dicdir -u new.dic /Users/koara/nlp-100-nock/tutrials/seeds/dict.csv

# 追加した語彙を使用して分ち書きする

t = MeCab.Tagger(r'-O wakati -u /Users/koara/nlp-100-nock/venv/lib/python3.8/site-packages/unidic_lite/dicdir/new.dic')
result = t.parse('明日は雨だよ、ぴえん。')
print(result.split())

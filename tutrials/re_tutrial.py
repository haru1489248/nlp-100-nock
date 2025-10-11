import re

# m = re.match(r'正規表現', 文字列) 先頭がマッチしていたらマッチオブジェクトを返す
# m = re.search(r'正規表現', 文字列) 文字列のどこかがマッチしていたらマッチオブジェクトを返す
# m.group() マッチした文字列を返す
# m.start() マッチした部分の開始インデックスを返す
# m.end() マッチした部分の終了インデックスを返す
# m.span() マッチした部分の開始インデックスと終了インデックスのタプルを返す

s1 = 'aa000'
m = re.match(r'[0-9]{3}', s1)

print(m)
if m:
    print(m.group())
    print(m.span())

s2 = 'bananaは¥300です'
m = re.search(r'¥[1-9][0-9]*', s2)

print(m)
if m:
    print(m.group())
    print(m.span())

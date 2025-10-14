import CaboCha
cp = CaboCha.Parser()
sentence = 'もし俺が謝ってこられてきてたとしたら、絶対に認められてたと思うか？'
tree = cp.parse(sentence)
print(tree.toString(CaboCha.FORMAT_XML))

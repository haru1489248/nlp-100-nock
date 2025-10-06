import string

s = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."

designated_indices = [1, 5, 6, 7, 8, 9, 15, 16, 19]

table = str.maketrans('', '', string.punctuation)
words = s.translate(table).split()

elem_map = {}
for i, word in enumerate(words, start=1):
    key = word[:1] if i in designated_indices else word[:2]
    elem_map[key] = i

print(elem_map)

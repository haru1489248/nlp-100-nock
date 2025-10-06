s1 = "paraparaparadise"
s2 = "paragraph"

def ngrams(seq, n: int):
    if n <= 0:
        raise ValueError("n must be >= 1")
    L = len(seq)
    return [seq[i:i+n] for i in range(L - n + 1)]

X = set(ngrams(s1, 2))
Y = set(ngrams(s2, 2))

print("和集合：", X | Y)
print("積集合：", X & Y)
print("差集合：", X - Y)
print("se ∈ X：","se" in X)
print("se ∈ Y：","se" in Y)

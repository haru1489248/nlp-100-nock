s = "I am an NLPer"
def ngrams(seq, n: int):
    if n <= 0:
        raise ValueError("n must be >= 1")
    L = len(seq)
    result = []
    for i in range(L - n + 1):
        gram = seq[i:i+n]
        result.append(gram)
    return result

print(ngrams(s, 2))
print(ngrams(s, 3))

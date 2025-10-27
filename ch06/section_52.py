from pathlib import Path
from gensim.models import KeyedVectors

src = Path('ch06/assets/GoogleNews-vectors-negative300.bin.gz')

s = 'United_States'

vectors = KeyedVectors.load_word2vec_format(
    src,
    binary=True
)

top10 = vectors.most_similar(s, topn=10)

for i, (word, score) in enumerate(top10, start=1):
    print(f"{i}位-{word}： {score}")

from pathlib import Path
from gensim.models import KeyedVectors
import numpy as np

src = Path('ch06/assets/GoogleNews-vectors-negative300.bin.gz')

vectors = KeyedVectors.load_word2vec_format(
    src,
    binary=True
)

w1, w2, w3 = 'Spain', 'Madrid', 'Athens'

spain_vector = vectors[w1]
madrid_vector = vectors[w2]
athens_vector = vectors[w3]

result_vector = spain_vector - madrid_vector + athens_vector

top10 = vectors.similar_by_vector(result_vector, topn=10)

for i, (word, score) in enumerate(top10, start=1):
    print(f"{i}位-{word}：{score}")

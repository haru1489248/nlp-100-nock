from pathlib import Path
from gensim.models import KeyedVectors

src = Path("ch06/assets/GoogleNews-vectors-negative300.bin.gz")
vectors = KeyedVectors.load_word2vec_format(src, binary=True)

if "United_States" in vectors:
    print(vectors["United_States"][:10])

from pathlib import Path
from gensim.models import KeyedVectors

src = Path('ch06/assets/questions-words.txt')
vector_src = Path('ch06/assets/GoogleNews-vectors-negative300.bin.gz')
target_section = "capital-common-countries"

vectors = KeyedVectors.load_word2vec_format(
    vector_src,
    binary=True
)

in_target = False

with src.open(encoding='utf-8') as f:
    for line in f:
        if not line:
            continue
        if line.startswith(':'):
            in_target = (line[1:].strip() == target_section)
            continue
        if not in_target:
            continue
        items = line.split()
        w1, w2, w3, w4_gold = items

        try:
            # vec(w2) - vec(w1) + vec(w3) に最も近い語を1件
            pred, score = vectors.most_similar(
                positive=[w2, w3], negative=[w1], topn=1
            )[0]
        except KeyError:
            continue
        print(f"{w2} - {w1} + {w3} => pred={pred} (score={score:.4f}), gold={w4_gold}")

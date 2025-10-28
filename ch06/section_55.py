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
gold_count = 0
count = 0

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
        if pred == w4_gold:
            gold_count += 1
        count += 1
print(f"正解率{gold_count/count}")

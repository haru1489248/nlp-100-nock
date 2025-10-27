from pathlib import Path
from gensim.models import KeyedVectors
import numpy as np

src = Path('ch06/assets/GoogleNews-vectors-negative300.bin.gz')

vectors = KeyedVectors.load_word2vec_format(
    src,
    binary=True,
)

w1, w2 = "United_States", "U.S."

v1 = vectors[w1]
v2 = vectors[w2]

# np.linalg は NumPy の線形代数（linear algebra）用サブモジュール
# np dot はドット積を計算する関数
cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
print(f"コサイン類似度: {cos_sim}")

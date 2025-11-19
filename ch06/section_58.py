"""Ward法（ウォード法）"""

from scipy.cluster.hierarchy import dendrogram, ward
from pathlib import Path
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt

model_src = Path('ch06/assets/GoogleNews-vectors-negative300.bin.gz')
data_src = Path('ch06/assets/questions-words.txt')

vectors = KeyedVectors.load_word2vec_format(model_src, binary=True)

target_section = ["capital-common-countries", "capital-world"]

data = []
for target in target_section:
    with open(data_src, 'r', encoding='utf-8') as f:
       for line in f:
           line = line.split()
           if line[1].startswith(target):
               current_section = line[1]
           elif line and current_section == target:
               if len(line) == 4:
                   for word in line:
                       data.append(vectors[word])

Z = ward(data)

dendrogram(Z)
plt.show()

# TODO: ユークリッド距離の勉強をする

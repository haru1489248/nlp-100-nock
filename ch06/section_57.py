from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns


model_src = Path('ch06/assets/GoogleNews-vectors-negative300.bin.gz')
data_src = Path('ch06/assets/questions-words.txt')

vectors = KeyedVectors.load_word2vec_format(model_src, binary=True)

data = []
target_section = ["capital-common-countries", "capital-world"]

for target in target_section:
    with open(data_src, 'r', encoding='utf-8') as f:
     for line in f:
        line = line.split()
        if line[1].startswith(target):
            current_section = line[1]
        elif line and current_section == target:
            words = line
            if len(words) == 4:
                data.append(words)

df = pd.DataFrame(data, columns=["word1", "word2", "Human (mean)", "word4"])

countries = pd.unique(df["word4"])

# KMeansなどは二次元配列を想定しているので組み立てる
X = []
for i in countries:
   X.append(vectors[i])
X = np.array(X)

# 5 class 分類
label = KMeans(n_clusters=5, random_state=20251030).fit_predict(X)

# グラフ表示のために次元圧縮（２次元）
reduced_coor = PCA(n_components=2).fit_transform(X)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

ax = sns.scatterplot(x=reduced_coor[:, 0], y=reduced_coor[:, 1], hue=label, palette="Set1")

for i, name in enumerate(countries):
   ax.annotate(name, (reduced_coor[i, 0], reduced_coor[i, 1]), xytext=(5, 5), textcoords='offset points')

plt.title("k-means clustering")
plt.tight_layout()
fig.savefig("k-means-countries.jpg")
plt.clf()
plt.close()

# TODO:　 KMeansのKの選定方法、PCAとは何か

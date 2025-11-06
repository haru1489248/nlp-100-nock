import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

df = pd.read_csv("ml/assets/sample.csv", index_col=0)

model_svd = TruncatedSVD(n_components=2) # n components で次元数指定
vecs_list = model_svd.fit_transform(df)
X = vecs_list[:, 0]
Y = vecs_list[:, 1]

plt.figure(figsize=(6,6))
sns.scatterplot(x=X, y=Y) # 散布図
plt.show()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing # bostonのデータセットは倫理的な問題で削除されたので使用不可

df = sns.load_dataset('titanic') # titanicはタイタニック号のデータ

sns.displot(df["age"]) # displotはヒストグラムの描写

sns.countplot(x="pclass", data=df) # pclassは乗客の階級のデータ
sns.catplot(x="pclass", data=df, kind='count') # catplotは汎用性が高い。上と同じ

sns.barplot(x='survived', y='age', hue='sex', data=df) # sexは性別、barplotは平均値を出力する
sns.catplot(x='survived', y='age', hue='sex', data=df, kind='bar') # 上と同じ
mean = df.groupby(["survived", "sex"])["age"].mean() # 上と同じ値を取得できる

sns.boxplot(x='survived', y='age', hue='sex', data=df) # 箱ひげ図
sns.catplot(x='survived', y='age', hue='sex', data=df, kind='box')

sns.violinplot(x='survived', y='age', hue='sex', data=df) # ヴァイオリンプロット、どこにボリュームゾーンがあるのかがわかりやすい。ほぼ箱ひげ図
sns.catplot(x='survived', y='age', hue='sex', data=df, kind='violin')

sns.jointplot(x='age', y='fare', data=df) # xとyの散布図、ヒストグラムを表示

df = sns.load_dataset('iris')
sns.pairplot(df) # 散布図、ヒストグラムを表示,複数変数があっても可能

# 相関係数
print(df.corr(numeric_only=True)) # 数値のみ使用を明示的に指定しないといけない

california_housing = fetch_california_housing()
df = california_housing.data
df = pd.DataFrame(df, columns=california_housing.feature_names)
sns.heatmap(df.corr(numeric_only=True)) # ヒートマップ、どこに相関があるのかについてぱっと見でわかる
plt.show()

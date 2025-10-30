from gensim.models import KeyedVectors
import pandas as pd

human_evaluation_src = 'ch06/assets/combined.csv'
model_src = 'ch06/assets/GoogleNews-vectors-negative300.bin.gz'
puts_file = 'ch06/assets/sample.csv'

vectors = KeyedVectors.load_word2vec_format(model_src, binary=True)

df = pd.read_csv(human_evaluation_src)

df['word2vec_sim'] = df.apply(lambda row: vectors.similarity(row['Word 1'], row['Word 2']), axis=1)

corr_spearman = df[["Human (mean)", "word2vec_sim"]].corr(method="spearman")

print(corr_spearman.iloc[0, 1])

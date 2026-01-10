import os
import pandas as pd
from pathlib import Path

dir = os.environ.get("ASSETS_DIR")
dir = Path(dir)

train_file = 'SST-2/train.tsv'
dev_file = 'SST-2/dev.tsv'

train_df = pd.read_csv(dir / train_file, sep='\t') #  / 演算子でパスを結合できるのは Path オブジェクトだけ
dev_df = pd.read_csv(dir / dev_file, sep='\t')

def sentence_to_tokens_with_count(sentence):
    tokens = sentence.split()
    feature = {}
    for token in tokens:
        if token in feature:
            feature[token] += 1
        else:
            feature[token] = 1
    return feature

def row_to_vec(row):
    return {
        'text': row['sentence'],
        'label': row['label'],
        'feature': row['feature']
    }
# dfのapply関数はそれぞれの列（axis=１で行）に対して関数を実行できる関数
train_df['feature'] = train_df['sentence'].apply(sentence_to_tokens_with_count)
dev_df['feature'] = dev_df['sentence'].apply(sentence_to_tokens_with_count)

# tolist関数はpandasのSeriesやnumpyの配列をpythonの通常リストに変換するメソッド
train_vec = train_df.apply(row_to_vec, axis=1).tolist()
dev_vec = dev_df.apply(row_to_vec, axis=1).tolist()

# 学習データの最初の事例を目視で確認
print("学習データの最初の事例:")
print(train_vec[0])

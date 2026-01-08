import csv

dev_path = './ch07/assets/SST-2/dev.tsv'
train_path = './ch07/assets/SST-2/train.tsv'

with open(train_path, 'r', encoding='utf-8') as f:
    csv_reader = csv.DictReader(f, delimiter='\t')
    train_positive_count = 0
    train_negative_count = 0
    for row in csv_reader:
        if int(row['label']) == 0:
            train_negative_count += 1
        else:
            train_positive_count += 1
    print('===訓練データ（train）===')
    print(f"ポジティブ事例数：{train_positive_count}件")
    print(f"ネガティブ事例数：{train_negative_count}件")
    print('==================')

with open(dev_path, 'r', encoding='utf-8') as f:
    csv_reader = csv.DictReader(f, delimiter='\t')
    dev_positive_count = 0
    dev_negative_count = 0
    for row in csv_reader:
        if int(row['label']) == 0:
            dev_negative_count += 1
        else:
            dev_positive_count += 1
    print('===開発データ（dev）===')
    print(f"ポジティブ事例数：{dev_positive_count}件")
    print(f"ネガティブ事例数：{dev_negative_count}件")
    print('==================')


# 別解
import pandas as pd
import os

from pathlib import Path

dir = os.environ.get("ASSETS_DIR")
dir = Path(dir)

train_file = 'SST-2/train.tsv'
dev_file = 'SST-2/dev.tsv'

train_df = pd.read_csv(dir / train_file, sep='\t')
dev_df = pd.read_csv(dir / dev_file, sep='\t')

print(
    f"""
学習データ
ポジティブ:\t{(train_df["label"] == 1).sum()}
ネガディブ:\t{(train_df["label"] == 0).sum()}

検証データ
ポジティブ:\t{(dev_df["label"] == 1).sum()}
ネガディブ:\t{(dev_df["label"] == 0).sum()}
"""
)

"""モデルの学習.

CPU上で学習を行う。

1. Dataset(SST-2, train/dev)の読み込み
2. Datasetに含まれる語彙の取得
3. 単語埋め込み行列, key-index辞書の作成
4. Datasetの前処理(token->idに変換)
5. DataLoaderの作成
6. 学習

===================================
Refs:
1. Optimizerの解説記事
【決定版】スーパーわかりやすい最適化アルゴリズム -損失関数からAdamとニュートン法-
https://qiita.com/omiita/items/1735c1d048fe5f611f80
"""

import os
import random
from pathlib import Path
# UnionはAまたはBのどちらかの方が入る可能性がある場合に使用する（例:Union[str, int]->str or int）
from typing import Any, Dict, List, Set, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
# optimは勾配降下法などの最適化アルゴリズムをまとめているclass
from torch import optim
# Datasetとは、PyTorchのDataLoaderやSamplerなどが前提としている
# 「データ集合の共通インターフェース（__len__ と __getitem__）」を定義するための基底クラス
# DataLoaderとはDatasetを使いやすく管理するためのclass（詳しくは下に書いている:295）
from torch.utils.data import DataLoader, Dataset
# tqdmとは進捗確認ができるライブラリ
from tqdm import tqdm

class SSTDataset(Dataset):
    """
    Dataset Class for the SST-2.
    """

    def __init__(self, data: List[Dict[str, torch.Tensor]], embedding_matrix: torch.Tensor) -> None:
        super().__init__()
        self.data = data
        self.embedding_matrix = embedding_matrix

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        object = self.data[index]
        input_ids = object["input_ids"]
        embeddings = self.embedding_matrix[input_ids]

        # 平均化ベクトルの取得
        # torch.meanはtensorに含まれるすべての要素の平均を返す
        # dimを0と指定すると列方向に平均、1と指定すると行方向に平均をとる
        mean_embedding = torch.mean(embeddings, dim=0)
        return mean_embedding, object["label"]

class SemanticClassifier(nn.Module):
    """
    Bag of words.
    """

    def __init__(self, in_dimension: int, n_classes: int, device=None) -> None:
        super(SemanticClassifier, self).__init__()
        self.in_dimension = in_dimension
        self.n_classes = n_classes
        self.linear1 = nn.Linear(in_features=in_dimension, out_features=1, bias=False, device=device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # squeezeは、サイズが1の次元（軸）を削除して次元数を減らす関数
        return self.sigmoid(self.linear1(x)).squeeze(1)

# 乱数を使用するすべてのライブラリで同じ乱数を使用するために定義している
def fix_seeds(seed: int) -> None:
    """Fix seeds, Pytorch, random, numpy.

    Parameters
    ----------
    seed : int
        Number of a seed.
    """
    # random.random()・random.shuffle()などに影響する
    random.seed(seed)

    # おそらく間違いで書かれていたコード
    # これはseed付き乱数生成器を作っているだけで使っていない（インスタンス化している）
    # npのseedは使われていなかったので大丈夫だった
    np.random.RandomState(seed)
    #正解はグローバルなnumpyの乱数を固定するために定義したこれ
    np.random.seed(seed)

    # PyTorch（CPU）の乱数を固定する
    torch.manual_seed(seed)
    # GPUの場合
    torch.cuda.manual_seed(seed)

    # GPU内で同じ入力の場合に同じ出力を保証する設定
    # cuDNNはNVIDIAが提供するGPU用の高速数値計算ライブラリ
    # PyTorchでは内部でこれを使用してLinear、Conv、RNNなどを高速化している
    # 計算速度を多少遅くするが結果が必ず同じアルゴリズムだけを使用するようにできる
    torch.backends.cudnn.deterministic = True

def create_embedding_matrix(
    word_embedding_model_path: Union[str, Path],
    vocabulary: Set[str],
) -> Tuple[Dict[str, int], torch.Tensor]:
    """Extract a matrix from the pre-trained word embedding vector.

    Parameters
    ----------
    word_embedding_model_path : Union[str, Path]
        Path to the pre-trained word embedding model
    start_index : int, optional
        Starting index for the vocabulary, by default 0

    Returns
    -------
    Tuple[torch.Tensor, Dict[str, int]]
        Embedding matrix and word to index mapping

    Reference
    ---------
    https://github.com/upura/nlp100v2025/blob/main/ch08/ans73.py#L88C1-L91C57
    """
    wv_from_bin = KeyedVectors.load_word2vec_format(word_embedding_model_path, binary=True)

    # "<PAD>"は予約語
    key_to_idx = {"<PAD>": 0}

    # 単語埋め込み行列の取得
    # 最初の行は<PAD>用
    # wv_from_bin.vectorsで行列を取り出す（行が単語、列が埋め込み次元）
    # .shapeは(単語数, 埋め込み次元数)のtupleを返す
    _, d_emb = wv_from_bin.vectors.shape
    # E: List[Tensor]  （各 Tensor は shape=(d_emb,)）
    # 中身は
    # E = [
    #.   tensor([0.0, 0.0, 0.0, ..., 0.0]) 長さ300
    # ]
    E = [torch.zeros(d_emb, dtype=torch.float32)]

    # 単語が学習済み単語ベクトルに含まれているときのみ、ベクトルを取得
    # vocabularyは語彙のiterable
    # Setがなので集合で同じ要素を一回しか持てないデータ構造: {"a", "b", "c"}
    for word in vocabulary:
        # wv_from_bin.key_to_indexは対応する語彙の行のindexを返す
        # wv_from_bin[word]と直接するとなかった場合にKeyErrorが返ってくる可能性があるので存在チェックしている
        if word in wv_from_bin.key_to_index:
            # word を key、対応する行番号（index）を value に持つ辞書を作成する
            # len(key_to_idx) は現在の要素数なので、次に割り当てる index として正しい
            # （すでに <PAD> が index=0 として入っている前提）
            key_to_idx[word] = len(key_to_idx)

            # word に対応する埋め込みベクトル（1単語=1行）を取得し、
            # 埋め込み行列 E（リスト）の末尾に追加する
            E.append(torch.tensor(wv_from_bin[word]))

    # torch.stack()とは？
    # 同じ shape の Tensor を新しい次元を作って重ねるメソッド
    # E は (d_emb,) の Tensor を要素にもつ Python の list
    # 結果として (vocab_size, d_emb) の埋め込み行列になる
    embedding_matrix = torch.stack(E)

    return key_to_idx, embedding_matrix

def tokenize(row: pd.Series, key_to_idx: Dict[str, int]) -> Tuple[Dict[str, Any], int]:
    """Convert inputted text and label to dict object.

    Parameters
    ----------
    row : pd.Series
        Row of the dataset.
    key_to_idx : Dict[str, int]
        Dictionary of word to index.
    Returns
    -------
    Tuple[Dict[str, Any], int]
        Tokenized data dictionary and token count
    """
    sentence = row["sentence"]
    label = row["label"]
    input_ids = []

    for word in sentence.lower().split():
        if word in key_to_idx:
            input_ids.append(key_to_idx[word])

    token_dict = {
        "text": sentence,
        "label": torch.tensor(label, dtype=torch.long),
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
    }

    return token_dict, len(input_ids)

def convert_to_token(df: pd.DataFrame, key_to_idx: Dict[str, int]) -> List[Dict[str, torch.Tensor]]:
    """Apply tokenize function to each row of the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset dataframe.
    key_to_idx : Dict[str, int]
        Dictionary of word to index.

    Returns
    -------
    List[Dict[str, torch.Tensor]]
        List of tokenized data dictionaries
    """
    # sentenceのindex化
    # argsとはapplyで呼ばれる関数にrow以外で追加で渡したい引数を指定するもの
    # tupleで指定している理由
    # argsはtupleで渡さないといけないので(key_to_idx,)としている:(key_to_idx)だとint
    tokenized_data = df.apply(tokenize, args=(key_to_idx,), axis=1)

    # token数が0の行を除く
    #.token_count = input_idsで単語ベクトルが存在しないものをフィルターしている
    result = [token_dict for token_dict, token_count in tokenized_data if token_count > 0]

    return result

# sentenceは文章を要素に持つ配列
def get_vocabulary(sentence: List[str]) -> Set[str]:
    """Get the set of vocabulary in the dataset.

    Parameters
    ----------
    sentence : List[str]
        List of texts.

    Returns
    -------
    Set[str]
    """
    result = set()

    for item in sentence:
        result.update(item.lower().split())

    return result

def train(
    model: SemanticClassifier, # ロジスティック回帰モデル
    trainloader: DataLoader,
    devloader: DataLoader,
    optimizer: optim.Adam, # momentumとRMSPropを組み合わせた手法
    criterion: nn.BCELoss, # Binary Cross Entropy Loss（2値交差エントロピー損失）
    epoch: int, # current epoch
    epochs: int, # total epoch
    # torch.deviceはTensorをどこに載せるかを型安全に指定できるclassで
    # indexをつけることによって複数のGPUを設定することができる
    device: Union[str, torch.device] = "cpu",
) -> None:
    """Train the model.

    Parameters
    ----------
    model : SemanticClassifier
        Model to train.
    trainloader : DataLoader
        DataLoader for training.
    optimizer : optim.Adam
        Optimizer for training.
    criterion : nn.BCELoss
        Loss function for training.
    epoch : int
        Current epoch.
    epochs : int
        Total number of epochs.
    device : Union[str, torch.device], optional
        Device to use for training.
    """

    """
    nn.Module.train()の説明
        呼び出すとmodeがTrueにセットされる(true=training mode)
        mode=Trueで挙動が変わるもの
        - torch.nn.Dropout
            学習時に毎回特徴を一定の確率(p:引数に入れられる)で落とすという処理をして
            同じ特徴だけ学習される過学習を防ぐ
            評価時はDropoutの出力はそのままになる（特徴を落とさない）
        - torch.nn.BatchNorm2d
            詳しくはまた今度学習する（正規化に使う平均・分散などが絡んでいる内容だった）
    """
    model.train()

    # 学習の様子を可視化する為に用意
    total_loss = 0.0 # バッチの損失の合計
    num_batches = 0 # 足し合わせた回数

    # trainloader（DataLoader）は、Datasetを材料にして`__getitem__`と`__len__`を自動で呼びながら
    # バッチ単位のデータを返してくれる仕組み
    # DataLoaderがやっていること
    # 1. len(train_dataset)を呼んでデータ数を調べる
    # 2. indexのリストを作成する：[0, 1, 2, ..., N-1]
    # 3. shuffle=Trueならindexをshuffleする
    # 4. batch_sizeごとにindexを区切る
    # 5. 各indexに対してdataset[i]を呼ぶ
    # 6. 返ってきたデータをまとめて1バッチにする
    # 7. forループに1バッチずつ渡す
    # ※ dataloaderの instanceを作成するときにdataset, batch, shuffle(boolean)などを設定する
    with tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}") as t:
        # 1バッチ（batch_size件）ごとにmean_embedding, labelを取り出す
        for mean_embedding, label in t:
            # Tensorを指定したdevice（CPU/GPU）に移動する（同じdeviceの場合はそのまま返る）
            mean_embedding = mean_embedding.to(device)
            # BCELoss（やSigmoidの出力）は「確率」を扱うのでfloat32を指定している
            label = label.to(device).to(torch.float32)

            # optimizerの初期化
            # 前のバッチで計算された「勾配」を全て0にリセットする
            optimizer.zero_grad()

            # 推論
            # 内部でmodel.__call__()が呼ばれる
            # nn.Module.__call__()がforwardを呼び出すのでpredが返る
            pred = model(mean_embedding)

            # 損失値の算出
            # 損失は各バッチごとに算出される
            loss = criterion(pred, label)

            # 損失値を基にした勾配の計算
            # ここでは ∂loss/∂W（損失関数を重み行列で偏微分したもの）が
            # model.linear1.weight.grad に Tensor として格納される
            loss.backward()

            # 勾配を基にAdamアルゴリズムを用いて重み更新
            optimizer.step()

            # 損失値の記録
            # loss.item()でPyTorchのTensorをPythonのfloatに変換する
            total_loss += loss.item()
            num_batches += 1

            # バッチごとの損失を表示する
            t.set_postfix(train_loss=f"{loss.item():.4f}")

        # dev datasetでの評価
        # avg_train_lossで1エポックの平均損失を計算
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0
        dev_loss, dev_accuracy = evaluate(model, devloader, criterion, device)

        # t.set_postfixとは
        # プログレスバーの表示内容を更新する関数
        # 全体結果を表示している
        # .4fで小数点第4位まで表示
        t.set_postfix(train_loss=f"{avg_train_loss:.4f}", dev_loss=f"{dev_loss:.4f}", dev_acc=f"{dev_accuracy:.4f}")

    # エポック終了時の詳細表示
    print(f"\nEpoch {epoch + 1}/{epochs} Summary:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Dev Loss: {dev_loss:.4f}")
    print(f"  Dev Accuracy: {dev_accuracy:.4f} ({dev_accuracy * 100:.2f}%)")

def evaluate(
    model: SemanticClassifier,
    devloader: DataLoader,
    criterion: nn.BCELoss,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[float, float]:
    """Evaluate the model on dev dataset.

    Parameters
    ----------
    model : SemanticClassifier
        Model to evaluate.
    devloader : DataLoader
        DataLoader for evaluation.
    criterion : nn.BCELoss
        Loss function for evaluation.
    device : Union[str, torch.device], optional
        Device to use for evaluation.

    Returns
    -------
    Tuple[float, float]
        Average loss and accuracy on dev dataset.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    # torch.no_grad()とは？
    # 推論・評価用の処理なので、計算グラフを作らず、勾配を計算・追跡しないようにしている
    # 評価では学習しないため loss.backward() や optimizer.step() を呼ばない
    # 計算グラフを作成しないことで、メモリ使用量を抑え処理を高速化できる
    # 計算グラフ（computational graph）
    #
    # 順伝播で行った計算の依存関係を記録したDAG（有向・非巡回な関係）。
    # `requires_grad=True` のTensorを使うと自動で構築される。
    # `loss.backward()` でグラフを逆にたどり、勾配（∂loss/∂param）を計算して `param.grad` に入れる。
    # 学習時は中間値を保持するためメモリを使う。
    # 推論・評価では `torch.no_grad()` により計算グラフを作らず高速化できる。
    with torch.no_grad():
        for mean_embedding, label in devloader:
            mean_embedding = mean_embedding.to(device)
            label = label.to(device).to(torch.float32)

            pred = model(mean_embedding)
            loss = criterion(pred, label)

            # pred.squeeze() でサイズが 1 の次元を削除する（labelのshapeと揃えるため）
            # 今回は (batch_size=32, 1) の Tensor を (32,) にする
            # 値は変わらず、Tensor の shape だけが変わる
            # >= 0.5 で確率を True / False に二値化する
            # float() で bool Tensor (True/False) を 1.0 / 0.0 に変換する
            pred_binary = (pred.squeeze() >= 0.5).float()

            # pred_binaryとlabelのtensorが一致する件数を合計する
            # item()は0次元のTensorをPythonの数値（int or float）に変換する
            correct += (pred_binary == label).sum().item()

            # size(0)でindexが0番目の次元の長さを取得する（バッチ内の件数）
            total += label.size(0)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # 正解率の計算
    # 正解率 = 正解数 / 合計数
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy

def main(args) -> None:
    fix_seeds(args.seed)
    data_dir = Path(DATA_DIR)

    # 1. Datasetの読み込み
    train_df = pd.read_csv(data_dir / "SST-2/train.tsv", sep="\t")
    dev_df = pd.read_csv(data_dir / "SST-2/dev.tsv", sep="\t")

    # 2. Datasetに含まれる語彙の取得
    vocabulary = get_vocabulary(train_df["sentence"].tolist())

    # vocabulary.update()とは？
    # vocabularyはset型のstrを持つデータ構造
    # set.update(iterable)とすると他のiterableの要素を全部追加する（重複は自動で排除される）
    vocabulary.update(get_vocabulary(dev_df["sentence"].tolist()))

    # 3. 単語埋め込み行列, key-index辞書の作成
    key_to_idx, embedding_matrix = create_embedding_matrix(
        data_dir / "GoogleNews-vectors-negative300.bin.gz", vocabulary
    )

    # 4. Datasetの前処理(token->idに変換)
    train_data = convert_to_token(train_df, key_to_idx)
    dev_data = convert_to_token(dev_df, key_to_idx)

    train_dataset = SSTDataset(train_data, embedding_matrix)
    dev_dataset = SSTDataset(dev_data, embedding_matrix)

    if args.dryrun:
        print("dryrun. only 1 epoch.")
        epochs = 1
    else:
        epochs = args.epochs

    # 5. DataLoaderの作成
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    # 6. 学習
    device = torch.device("cpu")  # 本問題ではCPU上で学習する

    # embedding_matrix.size(1)で列の次元数を渡している（特徴量数）
    model = SemanticClassifier(in_dimension=embedding_matrix.size(1), n_classes=2, device=device)

    # torch.optim.Adamのinstanceを作成する
    # model.parameters()で学習で更新したパラメーター一覧（iterable)
    # 更新すべきパラメータをoptimizerに教えるために渡している
    # lrは学習率
    # betas=(b_1, b_2)はデフォルトで(0.9, 0.999)で設定される
    # betasは過去をどれだけ注視するかの値（notionに詳しく書いた）
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()

    for epoch in range(epochs):
        train(
            model=model,
            trainloader=train_loader,
            devloader=dev_loader,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            epochs=epochs,
            device=device,
        )

    # 学習済モデルの保存（第二引数に渡されたファイルに保存される）
    # 復元する時の例（今はあまり詳しく調べない）
    # model = SemanticClassifier(in_dimension=..., n_classes=2)
    # model.load_state_dict(torch.load("73_model.pth"))
    # model.eval()
    torch.save(model.state_dict(), "73_model.pth")


if __name__ == "__main__":
    # pythonスクリプトをコマンドラインから実行するときに渡された引数（args）を定義通りに解釈・解析（parse）するための# 標準ライブラリ
    import argparse

    # ArgumentParserのインスタンスを作成する（argumentを登録するために必要）
    parser = argparse.ArgumentParser()

    # add_argument(短い名前（省略可）, 長い名前, type・defaultなどの解釈ルール設定)
    parser.add_argument("-s", "--seed", type=int, default=29)
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("-b", "--batch_size", default=32, type=int)
    parser.add_argument("-p", "--postfix", type=str) # 今の所使用していない

    # store_trueはそのオプションが指定されたらTrue、指定されなかったらFalseにする
    parser.add_argument("--dryrun", action="store_true")

    # 内部でimport sysをしているのでimport不要
    # sys.argvを読んで登録したルールに従って解析をする
    # sys.argvの出力例
    # ['train.py', '-e', '50', '--dryrun']
    args = parser.parse_args()
    main(args)

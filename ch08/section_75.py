"""
75. パディング
複数の事例が与えられたときに、まとめてひとつのtensorオブジェクトで表現する関数`collate`を実装する。
与えられた複数の事例のトークン列の長さが異なるときは、トークン列の長さが最も長いものに揃え、0番トークンIDでパディングする。
トークン列の長さが長いものから順に、事例を並び替える。
"""

import torch
from typing import Dict, List, Union

def example_padding(input_ids: torch.Tensor, max_length: int, pad_value: int) -> torch.Tensor:
    current_len = input_ids.size(0)

    pad_size = max_length - current_len

    dtype = input_ids.dtype
    device = input_ids.device

    # torch.empty() は CPU / GPU 上にメモリ領域を確保するだけで、初期化は行わない
    # そのため Tensor の中身は未定義で、以前そのメモリに入っていた値が残っている可能性がある
    result = torch.empty(max_length, dtype=dtype, device=device)

    result[:current_len] = input_ids

    if pad_size > 0:
        result[current_len:] = pad_value

    return result

def chat_gpt_answer_padding(input_ids: torch.Tensor, max_length: int, pad_value: int) -> torch.Tensor:
    current_length = input_ids.size(0)
    # torch.full()は指定されたfill_valueを使用して大きさsize分埋めてtensorを返す
    # 基本形 torch.full(size, fill_value, *, dtype=None, device=None)
    base = torch.full((max_length,), pad_value, dtype=input_ids.dtype, device=input_ids.device)
    # current_lengthまでinput_idsで埋める
    base[:current_length] = input_ids
    return base

def padding(input_ids: torch.Tensor, max_length: int, pad_value: int) -> torch.Tensor:
    current_length = input_ids.size(0)
    base = torch.zeros(max_length, dtype=input_ids.dtype, device=input_ids.device)
    base[:current_length] = input_ids
    if current_length < max_length:
        base[current_length:] = pad_value
    return base

def collate(
    batch: List[Dict[str, Union[torch.Tensor, str]]], pad_value: int = 0
) -> Dict[str, torch.Tensor]:
    lengths = [len(item["input_ids"]) for item in batch]
    # keyはどのようにソートするかを関数で指定する引数
    # 今回は降順なのでinput_idsの長さが長いものから順番に並べられる
    # xはbatchの長さのrangeを順番に入れているのでbatch分のindexを作成する
    sorted_batch = sorted(
        batch, key=lambda x: len(x["input_ids"]), reverse=True
    )  # True = 降順指定
    max_length = max(lengths)

    padded_input_ids = torch.stack(
        [padding(item["input_ids"], max_length, pad_value) for item in sorted_batch]
    )

    # tensor.view(-1) は要素数を保ったまま強制的にフラット（1次元）にするメソッド
    # 例: torch.tensor([[1, 2, 3], [4, 5, 6]]).view(-1)
    #     -> tensor([1, 2, 3, 4, 5, 6])
    #
    # squeeze() は shape が 1 の次元だけを削除するメソッド
    # 上の例は shape が (2, 3) なので、squeeze() を呼んでも変化しない
    #
    # [0] で label を 0次元Tensor（スカラーTensor）にしている
    # 例: tensor([0.]) -> tensor(0.)
    labels = torch.stack([item["label"].view(-1)[0] for item in sorted_batch])

    return {"input_ids": padded_input_ids, "label": labels}

def main() -> None:

    input = [
        {
            "text": "hide new secretions from the parental units",
            "label": torch.tensor([0.0]),
            "input_ids": torch.tensor(
                [5785, 66, 113845, 18, 12, 15095, 1594], dtype=torch.long
            ),
        },
        {
            "text": "contains no wit , only labored gags",
            "label": torch.tensor([0.0]),
            "input_ids": torch.tensor(
                [3475, 87, 15888, 90, 27695, 42637], dtype=torch.long
            ),
        },
        {
            "text": "that loves its characters and communicates something rather beautiful about human nature",
            "label": torch.tensor([1.0]),
            "input_ids": torch.tensor(
                [4, 5053, 45, 3305, 31647, 348, 904, 2815, 47, 1276, 1964],
                dtype=torch.long,
            ),
        },
        {
            "text": "remains utterly satisfied to remain the same throughout",
            "label": torch.tensor([0.0]),
            "input_ids": torch.tensor(
                [987, 14528, 4941, 873, 12, 208, 898], dtype=torch.long
            ),
        },
    ]
    result = collate(input)
    print(result)


if __name__ == "__main__":
    main()

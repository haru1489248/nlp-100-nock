import torch
import torch.nn as nn

class BoWLogisticRegression(nn.Module):
    """
    Bag of Words
    """

    def __init__(self, in_dimension: int, n_classes: int, device=None) -> None:
        super().__init__() # 継承classを初期化する
        self.in_dimension = in_dimension
        self.n_classes = n_classes
        self.linear1 = nn.Linear(
            in_features=in_dimension,
            out_features=1,
            bias=False,
            device=device
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.linear1(x))

if __name__ == "__main__":
    # 構築したネットワークの確認
    print(BoWLogisticRegression(in_dimension=300, n_classes=2))



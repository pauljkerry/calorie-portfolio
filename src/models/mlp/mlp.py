# mlp.py

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128], dropout=0.2):
        """
        シンプルな多層パーセプトロン（MLP）
        
        Args:
            input_dim (int): 入力特徴量の数
            output_dim (int): 出力クラス数
            hidden_dims (list of int): 隠れ層のユニット数
            dropout (float): Dropout 率
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
from typing import Union, Tuple

import torch
from torch import nn
from torch.nn import functional as F

class SDNPool(nn.Module):
    def __init__(self, target_size: Union[int, Tuple[int, int]]):
        super().__init__()
        self._alpha = nn.Parameter(torch.rand(1))
        self._max_pool = nn.AdaptiveMaxPool2d(target_size) # jak to działa?
        self._avg_pool = nn.AdaptiveAvgPool2d(target_size) # jak to działa?

    def forward(self, x):
        avg_p = self._alpha * self._max_pool(x)
        max_p = (1 - self._alpha) * self._avg_pool(x)
        mixed = avg_p + max_p
        return mixed


class StandardHead(nn.Module):
    def __init__(self, in_channels: int, out_features: int, pool_size: int = 4):
        super().__init__()
        self.out_features = out_features
        self._pooling = SDNPool(pool_size)
        self._fc = nn.Linear(in_channels * pool_size ** 2, out_features)

    def forward(self, x: torch.Tensor):
        x = F.relu(x)
        x = self._pooling(x)
        x = x.view(x.size(0), -1)
        x = self._fc(x)
        return x



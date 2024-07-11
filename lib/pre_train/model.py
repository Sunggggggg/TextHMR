import torch
import torch.nn as nn

from .STFormer import STFormer
from .CrossAtten import CrossAttentionBlock

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, ):
        return
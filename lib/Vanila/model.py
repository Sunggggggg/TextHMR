import torch
import torch.nn as nn
from .transformer import Transformer
from .regressor import Regressor

class Model(nn.Module):
    def __init__(self, embed_dim=512) :
        super().__init__()
        self.proj_input = nn.Linear(2048, embed_dim)
        self.proj_output = nn.Linear(embed_dim, 2048)


    def forward(self, f_img, is_train=False, J_regressor=None):
        
        

        return
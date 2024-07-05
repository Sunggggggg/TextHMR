import torch
import torch.nn as nn
from .transformer import CrossAttentionBlock

class Linker(nn.Module) :
    def __init__(self, 
                 embed_dim,
                 ) :
        super().__init__()
        self.joint_embedding = nn.Linear(3, embed_dim)

        self.motion2text = CrossAttentionBlock()

    def forward(self, pose3d, text_embed):
        """
        pose3d      : [B, J, 3]
        text_embed  : [B, N, dim]
        """
        self.joint_embedding(pose3d)

        return

def get_model():

    return 
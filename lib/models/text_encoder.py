import torch
import torch.nn as nn
from .transformer import Transformer

class TEncoder(nn.Module):
    def __init__(self, depth=3, embed_dim=512, mlp_hidden_dim=1024,
            h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=36):
        super().__init__()
        self.input_proj = nn.Linear(768, embed_dim)
        self.transformer = Transformer(depth=depth, length=length, embed_dim=embed_dim, mlp_hidden_dim=mlp_hidden_dim, h=h, 
                        drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate)
    
    def forward(self, text_embed):
        feature = self.input_proj(text_embed)
        feature = self.transformer(feature)

        return feature

class LQTEncoder(nn.Module):
    def __init__(self, depth=3, embed_dim=512, mlp_hidden_dim=1024,
            h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=36):
        super().__init__()
        num_queries = 16
        self.input_proj = nn.Linear(768, embed_dim)
        self.motion_query = nn.Parameter(torch.zeros((1, num_queries, embed_dim)))

        
        self.transformer = Transformer(depth=depth, length=length, embed_dim=embed_dim, mlp_hidden_dim=mlp_hidden_dim, h=h, 
                        drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate)
    
    def forward(self, text_embed):
        feature = self.input_proj(text_embed)
        feature = self.transformer(feature)

        return feature

def get_model(depth=3, length=16, embed_dim=512, mlp_hidden_dim=1024, h=8, drop_rate=0.2, drop_path_rate=0.2, attn_drop_rate=0.):
    model = TEncoder(depth=depth, length=length, embed_dim=embed_dim, mlp_hidden_dim=mlp_hidden_dim, h=h, 
                        drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate)
    return model
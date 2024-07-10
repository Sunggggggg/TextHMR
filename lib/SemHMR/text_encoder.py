import torch
import torch.nn as nn
from .transformer import Transformer

class CrossAttention(nn.Module):
    def __init__(self, dim, v_dim, kv_num, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.kv_num = kv_num
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(v_dim, v_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(v_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv):

        B, N, C = xq.shape
        v_dim = xv.shape[-1]
        q = self.wq(xq).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  
        k = self.wk(xk).reshape(B, self.kv_num, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  
        v = self.wv(xv).reshape(B, self.kv_num, self.num_heads, v_dim // self.num_heads).permute(0, 2, 1, 3) 

        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, v_dim) 
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

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

def get_model(depth=2, length=16, embed_dim=512, mlp_hidden_dim=1024, h=8, drop_rate=0.2, drop_path_rate=0.2, attn_drop_rate=0.):
    model = TEncoder(depth=depth, length=length, embed_dim=embed_dim, mlp_hidden_dim=mlp_hidden_dim, h=h, 
                            drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate)
    return model
import torch
import torch.nn as nn
from .transformer import Transformer

class CrossAttention(nn.Module):
    def __init__(self, dim, v_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
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
        q = self.wq(xq) # [B, Q, dim]
        k = self.wk(xk) # [B, K, dim]
        v = self.wv(xv) # [B, V, dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) # [B, Q, K] 

        x = (attn @ v).reshape(B, N, v_dim)     # [B, Q, K] @ [B, V, dim]
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
        #self.pool = nn.MaxPool2d((length, 1))
    
    def forward(self, text_embed):
        feature = self.input_proj(text_embed)
        feature = self.transformer(feature)     # [B, N, 128]
        #feature = self.pool(feature)            # [B, 1, 128]

        return feature

class LQTEncoder(nn.Module):
    def __init__(self, depth=3, embed_dim=512, mlp_hidden_dim=1024,
            h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=36):
        super().__init__()
        num_queries = 64
        self.input_proj = nn.Linear(768, embed_dim)
        self.motion_query = nn.Parameter(torch.zeros((1, num_queries, embed_dim)))
        self.motion_extract = CrossAttention(embed_dim, embed_dim, attn_drop=attn_drop_rate, proj_drop=drop_rate)

        self.transformer = Transformer(depth=depth, length=num_queries, embed_dim=embed_dim, mlp_hidden_dim=mlp_hidden_dim, h=h, 
                        drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate)
    
    def forward(self, text_embed, caption_len):
        """
        caption_len : [B]
        """
        B = text_embed.shape[0]
        feature = self.input_proj(text_embed)           # [B, 36, dim]

        motion_queries = []
        for b in range(B) :
            # Remove padding
            batch_text_embed = feature[b:b+1, :caption_len[b]]    # [1, n, dim]
            motion_query = self.motion_extract(motion_query, batch_text_embed, batch_text_embed)
            motion_queries.append(motion_query)
        motion_queries = torch.cat(motion_queries, dim=0)
        motion_queries = self.transformer(motion_queries)
        return motion_queries   # [B, 100, dim]

def get_model(learn_query=False, depth=2, length=16, embed_dim=128, mlp_hidden_dim=1024, h=8, drop_rate=0.2, drop_path_rate=0.2, attn_drop_rate=0.):
    if learn_query :
        model = LQTEncoder(depth=depth, length=length, embed_dim=embed_dim, mlp_hidden_dim=mlp_hidden_dim, h=h, 
                            drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate)
    else:
        model = TEncoder(depth=depth, length=length, embed_dim=embed_dim, mlp_hidden_dim=mlp_hidden_dim, h=h, 
                            drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate)
    return model
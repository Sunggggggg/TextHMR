import torch
import torch.nn as nn
from .transformer import Mlp
from timm.models.layers import DropPath

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
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.normq = norm_layer(dim)
        self.normk = norm_layer(dim)
        self.normv = norm_layer(dim)

        self.attn = CrossAttention(dim, dim, dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, xq, xk, xv):

        xq = xq + self.drop_path(self.attn(self.normq(xq), self.normk(xk), self.normv(xv)))
        xq = xq + self.drop_path(self.mlp(self.norm2(xq)))
        return xq

class Linker(nn.Module) :
    def __init__(self, 
                 embed_dim,
                 ) :
        super().__init__()
        self.joint_embedding = nn.Linear(3, embed_dim)

        self.joint_pos_embedding = nn.Parameter(torch.randn(1, 19, embed_dim))
        self.text_pos_embedding = nn.Parameter(torch.randn(1, 36, embed_dim))

        self.joint_query_pos_embedding = nn.Parameter(torch.randn(1, 19, embed_dim))
        self.text_query_pos_embedding = nn.Parameter(torch.randn(1, 36, embed_dim))
        self.joint_key_pos_embedding = nn.Parameter(torch.randn(1, 19, embed_dim))
        self.text_key_pos_embedding = nn.Parameter(torch.randn(1, 36, embed_dim))

        self.motion2text = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, pose3d, text_embed):
        """
        pose3d      : [B, J, 3]
        text_embed  : [B, N, dim]
        """
        pose_feat = self.joint_embedding(pose3d)


        return

def get_model():

    return 
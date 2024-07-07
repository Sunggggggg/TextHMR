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
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_kv, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.normq = norm_layer(dim)
        self.normk = norm_layer(dim)
        self.normv = norm_layer(dim)

        self.attn = CrossAttention(dim, dim, num_kv, num_heads=num_heads, qkv_bias=qkv_bias,
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
                 num_joints,
                 num_words,
                 num_heads=8,
                 depth=3,
                 ) :
        super().__init__()
        self.depth = depth

        self.joint_embedding = nn.Linear(3, embed_dim)

        self.joint_pos_embedding = nn.Parameter(torch.randn(1, num_joints, embed_dim))
        self.text_pos_embedding = nn.Parameter(torch.randn(1, num_words, embed_dim))

        self.joint_query_pos_embed = nn.Parameter(torch.randn(1, num_joints, embed_dim))
        self.text_query_pos_embed = nn.Parameter(torch.randn(1, num_words, embed_dim))
        self.joint_key_pos_embed = nn.Parameter(torch.randn(1, num_joints, embed_dim))
        self.text_key_pos_embed = nn.Parameter(torch.randn(1, num_words, embed_dim))

        self.motion2text_linker = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_words, num_heads, embed_dim*4.)
            for _ in range(depth)])
        self.text2motion_linker = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_joints, num_heads, embed_dim*4.)
            for _ in range(depth)])
        
        self.joint_head = nn.Sequential(
            nn.LayerNorm(embed_dim), 
            nn.Linear(embed_dim, 3)
        )
        self.text_head = nn.LayerNorm(embed_dim)


    def forward(self, pose3d, text_embed):
        """
        pose3d      : [B, J, 3]
        text_embed  : [B, N, dim]

        return 
        joint_guide     : [B, J, 3]
        semantic_guide  : [B, N, dim]
        """
        # Positional enmedding
        pose_feat = self.joint_embedding(pose3d)
        pose_feat = pose_feat + self.joint_pos_embedding
        text_feat = text_embed + self.text_pos_embedding

        # Cross Atten.
        joint_guide, semantic_guide = pose_feat, text_feat
        
        for i in range(self.depth):
            joint_guide = self.motion2text_linker[i](joint_guide + self.joint_query_pos_embed,
                                    text_feat + self.text_key_pos_embed, text_feat)
            semantic_guide = self.text2motion_linker[i](semantic_guide + self.text_query_pos_embed, 
                                    pose_feat + self.joint_key_pos_embed, pose_feat)
        
        pose_feat = pose3d + self.joint_head(joint_guide)
        semantic_guide = self.text_head(text_feat + semantic_guide)
        
        return joint_guide, semantic_guide

def get_model(embed_dim=256, num_joints=19, num_words=36, num_heads=8, depth=3):
    model = Linker(embed_dim=embed_dim, num_joints=num_joints, num_words=num_words, num_heads=num_heads, depth=depth)
    return model
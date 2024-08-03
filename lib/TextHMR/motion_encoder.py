import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from functools import partial
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp, Attention

from .operation.Part_refine import PartAttention_v2
from .operation.CrossAtten import CoTransformer
from .operation.text_encoder import TEncoder

class Model(nn.Module):
    def __init__(self, num_total_motion, num_frames=16, num_joints=17, num_words=36, embed_dim=256, depth=10) :
        super().__init__()
        self.mid_frame = 8
        self.num_words = num_words
        self.seqlen = num_frames
        self.part_atten = PartAttention_v2(depth=depth, embed_dim=embed_dim, mlp_hidden_dim=embed_dim*4., 
                 h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0.1, num_joints=num_joints, num_frames=num_frames)
        
        self.text_encoder = TEncoder(depth=2, embed_dim=embed_dim, mlp_hidden_dim=embed_dim*4., 
                                     h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=36)
        
        self.co_former = CoTransformer(seqlen=num_frames, num_joints=num_joints, num_words=36 ,embed_dim=embed_dim)
        
        self.joint_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3)
        )
    
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim//2)
        self.drop = nn.Dropout()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(embed_dim//2 * num_joints, num_total_motion)
        
    def text_prediction(self, x):
        """ Text predicting via joint features
        joint_feat : [B, T, J, dim]
        """
        x = self.norm(x)
        x = self.fc1(x)
        x = self.drop(x)
        x = x.flatten(-2)
        x = self.fc2(x)
        return x
    
    def extraction_features(self, pose_2d, text_embeds):
        """
        pose_2d         : [B, T, J, 3]
        text_embeds     : [7693]
        """
        # Stage 1
        joint_feat = self.part_atten(pose_2d)  # [B, T, J, dim] 
        pred_text = self.text_prediction(joint_feat)              # [B, num_total_motion]
        max_pred_text = torch.argmax(pred_text, dim=-1)           # [B]
        
        # Padding
        text_emb, caption_mask = [], []
        for idx in max_pred_text:
            motion_feat = torch.tensor(text_embeds[idx][0])                  # [N, 768]
            n = motion_feat.shape[0]
            # Padding
            motion_feat = torch.cat([motion_feat] + [torch.zeros_like(motion_feat[0:1]) for _ in range(self.num_words-n)], dim=0)
            mask = torch.ones((self.num_words), device=pose_2d.device)
            mask[:n] = 0.

            text_emb.append(motion_feat)
            caption_mask.append(mask)                # n

        text_emb = torch.stack(text_emb, dim=0).float().cuda()              # [B, N, 768]
        caption_mask = torch.stack(caption_mask, dim=0).bool().cuda()       # [B, N]

        #
        text_feat = self.text_encoder(text_emb, caption_mask)               # [B, N, dim]
        joint_feat = self.co_former(joint_feat, text_feat, caption_mask)    # [B, T, J, dim]             
        return joint_feat
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class GraphormerNet(nn.Module):
    def __init__(self, num_frames=16, num_joints=17, embed_dim=256, depth=10, num_heads=8, mlp_ratio=2., 
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, pretrained=False):
        super().__init__()

        in_dim = 3
        out_dim = 3    
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.joint_embed = nn.Linear(in_dim, embed_dim)
        #self.imgfeat_embed = nn.Linear(2048, embed_dim)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.depth = depth

        self.SpatialBlocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.TemporalBlocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm_s = norm_layer(embed_dim)
        self.norm_t = norm_layer(embed_dim)

        self.regression = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )
        self.fusion = torch.nn.Conv2d(in_channels=num_frames, out_channels=1, kernel_size=1)

    def SpaTemHead(self, x):
        b, t, j, c = x.shape
        x = rearrange(x, 'b t j c  -> (b t) j c')
        x = self.joint_embed(x)
        x += self.spatial_pos_embed
        x = self.pos_drop(x)
        spablock = self.SpatialBlocks[0]
        x = spablock(x)
        x = self.norm_s(x)
        
        x = rearrange(x, '(b t) j c -> (b j) t c', t=t)
        x += self.temporal_pos_embed
        x = self.pos_drop(x)
        temblock = self.TemporalBlocks[0]
        x = temblock(x)
        x = self.norm_t(x)
        return x

    def forward(self, x, return_joint=False):
        b, t, j, c = x.shape
        x = self.SpaTemHead(x) # bj t c
        
        for i in range(1, self.depth):
            SpaAtten = self.SpatialBlocks[i]
            TemAtten = self.TemporalBlocks[i]
            x = rearrange(x, '(b j) t c -> (b t) j c', j=j)
            x = SpaAtten(x)
            x = self.norm_s(x)
            x = rearrange(x, '(b t) j c -> (b j) t c', t=t)
            x = TemAtten(x)
            x = self.norm_t(x)

        x = rearrange(x, '(b j) t c -> b t j c', j=j)
        if return_joint :
            x = self.regression(x) # (b t (j * 3))
            x = x.view(b, t, j, -1)
            return x
        else :
            x = x.view(b, t, j, -1)
            return x
        #x = self.fusion(x).reshape(b, t, j, -1)

        
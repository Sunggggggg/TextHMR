import os
import torch
import torch.nn as nn
from functools import partial

from .DSTformer import DSTformer
from .transformer import Transformer
from .regressor import Regressor

class Model(nn.Module):
    def __init__(self, 
                 chk_filename='/mnt/SKY/MotionBERT/checkpoint/latest_epoch.bin',
                 pretrained_freeze=True) :
        super().__init__()
        model_backbone = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512, 
                                   depth=5, num_heads=8, mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                   maxlen=64, num_joints=17)
        self.proj_pos = nn.Linear(512, 128)
        self.proj_img = nn.Linear(2048, 512)
        self.temp_encoder = Transformer(depth=3, embed_dim=512, mlp_hidden_dim=512*4, 
                                        h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=64)
        self.proj_dec = nn.Linear(512+17*128, 256)
        self.temp_decoder = Transformer(depth=1, embed_dim=256, mlp_hidden_dim=256*4, 
                                        h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=64)
        self.regressor = Regressor(256)

        if chk_filename:
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
            
            if pretrained_freeze :
                for p in model_backbone.parameters():
                    p.requires_grad = False

        self.model_backbone = model_backbone

    def forward(self, f_img, pose_2d, is_train=False, J_regressor=None):
        B, T = pose_2d.shape[:2]

        pose_feat = self.model_backbone.get_representation(pose_2d)     # [B, T, J, 512]
        pose_feat = self.proj_pos(pose_feat)
        pose_feat = pose_feat.flatten(-2)                               # [B, T, J*128]
        
        img_feat = self.proj_img(f_img)                     
        img_feat = self.temp_encoder(img_feat)                          # [B, T, 512]

        feat = torch.cat([img_feat, pose_feat], dim=-1)                 # [B, T, 512+2176]
        feat = self.proj_dec(feat)
        feat = self.temp_decoder(feat)                  # [B, T, 256]

        if is_train:
            global_feat = feat
        else :
            global_feat = feat[:, T//2 : T//2 + 1]

        global_smpl_output, _ = self.regressor(global_feat, is_train=is_train, J_regressor=J_regressor)

        if not is_train:
            for s in global_smpl_output:
                s['theta'] = s['theta'].reshape(B, -1)
                s['verts'] = s['verts'].reshape(B, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(B, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, -1, 3, 3)

        else:
            size = T
            for s in global_smpl_output:
                s['theta'] = s['theta'].reshape(B, size, -1)
                s['verts'] = s['verts'].reshape(B, size, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(B, size, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, size, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, size, -1, 3, 3)

        return global_smpl_output, None




        return
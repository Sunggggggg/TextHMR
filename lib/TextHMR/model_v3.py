import os
import torch
import torch.nn as nn
import pandas as pd

from .smpl_regressor import SMPLRegressor
from .transformer import Transformer
from lib.pre_train.model_part import Model as pre_trained_model

class Model(nn.Module):
    def __init__(self, 
                 seqlen,
                 num_total_motion,
                 text_archive,
                 pretrained,
                 pretrained_freeze=True) :
        super().__init__()
        self.seqlen = seqlen
        self.text_archive = text_archive

        self.proj_img1 = nn.Linear(2048, 512)
        self.proj_img2 = nn.Linear(512, 256)
        self.temp_encoder = Transformer(depth=3, embed_dim=512, mlp_hidden_dim=2048, 
                                        h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=64)
        self.pre_trained_model = pre_trained_model(num_total_motion, seqlen)
        self.regressor = SMPLRegressor(dim_rep=512)

        if pretrained :
            pretrained_dict = torch.load(pretrained)['gen_state_dict']

            self.pre_trained_model.load_state_dict(pretrained_dict)
            print(f'=> loaded pretrained model from \'{pretrained}\'')
        
        if pretrained_freeze :
            for p in self.pre_trained_model.parameters():
                p.requires_grad = False
    
    def forward(self, f_img, pose_2d, is_train=False, J_regressor=None):
        """
        pose_2d     : [B, T, J, 3]
        img_feat    : [B, T, 2048]
        """
        B, T = f_img.shape[:2]
        pose_2d = pose_2d[..., :2]

        joint_feat = self.pre_trained_model.extraction_features(pose_2d, self.text_archive)   # [B, T, J, 256]
        
        img_feat = self.proj_img1(f_img)                     
        img_feat = self.temp_encoder(img_feat)              # [B, T, 512]
        img_feat = self.proj_img2(img_feat)                 # [B, T, 256]

        if is_train:
            global_joint_feat = joint_feat
            global_img_feat = img_feat
        else :
            global_joint_feat = joint_feat[:, T//2 : T//2 + 1]
            global_img_feat = img_feat[:, T//2 : T//2 + 1]

        global_smpl_output = self.regressor(global_joint_feat, global_img_feat, is_train=is_train, J_regressor=J_regressor)

        if not is_train:
            for s in global_smpl_output:
                s['theta'] = s['theta'].reshape(B, -1)
                s['verts'] = s['verts'].reshape(B, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(B, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, -1, 3, 3)

        else:
            size = self.seqlen
            for s in global_smpl_output:
                s['theta'] = s['theta'].reshape(B, size, -1)
                s['verts'] = s['verts'].reshape(B, size, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(B, size, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, size, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, size, -1, 3, 3)

        return global_smpl_output, None
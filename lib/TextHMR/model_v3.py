import os
import torch
import torch.nn as nn
import pandas as pd

from .regressor import Regressor
from .transformer import Transformer
from lib.pre_train.model_part import Model as pre_trained_model

class Model(nn.Module):
    def __init__(self, 
                 seqlen,
                 num_total_motion,
                 text_archive,
                 pretrained) :
        super().__init__()
        self.seqlen = seqlen
        self.text_archive = text_archive

        self.proj_img = nn.Linear(2048, 512)
        self.proj_joint = nn.Linear(256, 32)

        self.temp_encoder = Transformer(depth=3, embed_dim=512, mlp_hidden_dim=2048, 
                                        h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=64)
        self.pre_trained_model = pre_trained_model(num_total_motion, seqlen)

        self.fusing = nn.Sequential(nn.Linear(512+32*17, 512),
                                    nn.ReLU(), 
                                    nn.Linear(512, 512),
                                    nn.ReLU())
        self.global_regressor = Regressor(512)

        self.proj_local = nn.Sequential(nn.Linear(512+32*17, 512),
                                        nn.Linear(512, 256),
                                        nn.LayerNorm(256))
        self.local_encoder = Transformer(depth=2, embed_dim=256, mlp_hidden_dim=1024, 
                                        h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=9)
        self.local_regressor = Regressor(256)

        if pretrained :
            pretrained_dict = torch.load(pretrained)['gen_state_dict']

            self.pre_trained_model.load_state_dict(pretrained_dict)
            print(f'=> loaded pretrained model from \'{pretrained}\'')
    
    def forward(self, f_img, pose_2d, is_train=False, J_regressor=None):
        """
        pose_2d     : [B, T, J, 3]
        img_feat    : [B, T, 2048]
        """
        B, T = f_img.shape[:2]
        pose_2d = pose_2d[..., :2]

        joint_feat = self.pre_trained_model.extraction_features(pose_2d, self.text_archive)   # [B, T, J, 256]
        joint_feat = self.proj_joint(joint_feat)
        joint_feat = joint_feat.flatten(-2)                 # [B, T, 544]

        img_feat = self.proj_img(f_img)                     
        img_feat = self.temp_encoder(img_feat)             # [B, T, 512]
        f = torch.cat([joint_feat, img_feat], dim=-1)      # [B, T, 512+544]

        if is_train:
            global_feat = self.fusing(f)
        else :
            global_feat = self.fusing(f[:, T//2 : T//2 + 1])

        global_smpl_output, (pred_pose, pred_shape, pred_cam) = self.global_regressor(global_feat, is_train=is_train, J_regressor=J_regressor)

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

        mid_frame = T//2
        local_feat = self.proj_local(f[:, mid_frame-4:mid_frame+5]) # [B, T, dim]
        local_feat = self.local_encoder(local_feat)
        if is_train:
            local_feat = local_feat
        else :
            local_feat = local_feat[:, 4:5]

        smpl_output, _ = self.local_regressor(local_feat, pred_pose, pred_shape, pred_cam)

        if not is_train:
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(B, -1)
                s['verts'] = s['verts'].reshape(B, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(B, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, -1, 3, 3)

        else:
            size = 9
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(B, size, -1)
                s['verts'] = s['verts'].reshape(B, size, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(B, size, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, size, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, size, -1, 3, 3)

        return global_smpl_output, smpl_output, None
import os
import torch
import torch.nn as nn
from lib.core.config import BASE_DATA_DIR

import lib.SemHMR.temporal_encoder as temporal_encoder
import lib.SemHMR.text_encoder as text_encoder
import lib.SemHMR.highlight as highlight
import lib.SemHMR.transformer as transformer

from .HSCR import HSCR

class Model(nn.Module):
    def __init__(self,) :
        super().__init__()
        self.stride = 4
        self.init_hmr = temporal_encoder.get_model(depth=3, length=16, embed_dim=512)
        self.text_encoder = text_encoder.get_model(length=36, embed_dim=512)
        self.highlighter = highlight.get_model(embed_dim=512)

        self.proj_local1 = nn.Linear(2048, 256)
        self.proj_local2 = nn.Linear(512, 256)
        self.proj_local3 = nn.Linear(512, 256)
        self.local_trans = transformer.get_model(embed_dim=256, mlp_hidden_dim=256*4, length=self.stride*2+1)
        self.cross_atten1 = transformer.get_model_CA(embed_dim=256, kv_num=16)
        self.cross_atten2 = transformer.get_model_CA(embed_dim=256, kv_num=4)

        self.local_regressor = HSCR()
        
    def forward(self, input_text, img_feat, pose2d, caption_len, is_train=False, J_regressor=None):
        """
        input_text  : [B, 36, 768] 
        caption_len : [B]
        """
        B, T = img_feat.shape[:2]
        
        # First stage
        init_smpl_output, init_pred, temp_feat = self.init_hmr(img_feat, is_train, J_regressor)     # [B, T, *]
        text_embed = self.text_encoder(input_text)                                                  # [B, 1, 128]
        selected_text_embeds, batch_matrix = self.highlighter(temp_feat, text_embed, caption_len)   # [B, 4, 512]

        # Second stage
        local_feat = img_feat[:, T//2-self.stride : T//2+self.stride+1]
        local_feat = self.proj_local1(local_feat)
        local_feat = self.local_trans(local_feat)
        local_feat = local_feat[:, self.stride-1:self.stride+2]             # [B, 3, 256]

        temp_feat = self.proj_local2(temp_feat)                             # [B, T, 256]
        local_feat = self.cross_atten1(local_feat, temp_feat, temp_feat)

        selected_text_embeds = self.proj_local3(selected_text_embeds)
        local_feat = self.cross_atten2(local_feat, selected_text_embeds, selected_text_embeds)

        # Regressor
        if is_train:
            feature = local_feat
        else:
            feature = local_feat[:, 1][:, None, :]

        smpl_output = self.local_regressor(feature, init_pose=init_pred[0], init_shape=init_pred[1], init_cam=init_pred[2], is_train=is_train, J_regressor=J_regressor)
        
        scores = None
        if not is_train:
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(B, -1)
                s['verts'] = s['verts'].reshape(B, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(B, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, -1, 3, 3)
                s['scores'] = scores

        else:
            size = 3
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(B, size, -1)
                s['verts'] = s['verts'].reshape(B, size, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(B, size, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, size, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, size, -1, 3, 3)
                s['scores'] = scores

        return smpl_output, init_smpl_output, batch_matrix

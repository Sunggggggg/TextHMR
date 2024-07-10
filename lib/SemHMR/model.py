import os
import torch
import torch.nn as nn
from lib.core.config import BASE_DATA_DIR

import lib.SemHMR.temporal_encoder as temporal_encoder
import lib.SemHMR.text_encoder as text_encoder
import lib.SemHMR.highlight as highlight
import lib.SemHMR.transformer as transformer

class Model(nn.Module):
    def __init__(self,) :
        super().__init__()
        self.stride = 4
        self.init_hmr = temporal_encoder.get_model(depth=3, length=16, embed_dim=512)
        self.text_encoder = text_encoder.get_model(length=36, embed_dim=512)
        self.highlighter = highlight.get_model(embed_dim=512, stride=1)

        self.proj_local = nn.Linear(512, 256)
        self.proj_word = nn.Linear(36, 256)
        self.local_trans = transformer.get_model(embed_dim=256, mlp_hidden_dim=256*4, length=self.stride*2+1)
        
    def forward(self, input_text, img_feat, pose2d, is_train=False, J_regressor=None):
        """
        input_text : [B, 36, 768] 
        """
        B, T = img_feat.shape[:2]
        
        # First stage
        init_smpl_output, temp_feat = self.init_hmr(img_feat, is_train, J_regressor) # [B, T, *]
        text_embed = self.text_encoder(input_text)
        score_map = self.highlighter(temp_feat, text_embed)     # [B, 3, N]

        # Second stage
        local_feat = img_feat[:, T//2-self.stride : T//2+self.stride+1]
        local_feat = self.proj_local(local_feat)
        local_feat = self.local_trans(local_feat)
        local_feat = local_feat[:, self.stride-1:self.stride+2] # [B, 3, 256]

        self.proj_word(score_map)





        return init_smpl_output, score_map

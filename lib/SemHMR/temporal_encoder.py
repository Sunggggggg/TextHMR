import os
import torch
import torch.nn as nn
from lib.core.config import BASE_DATA_DIR

from .transformer import Transformer
from .spin import Regressor

class THMR(nn.Module):
    def __init__(self, 
                 depth=3, 
                 seqlen=16, 
                 embed_dim=512, 
                 h=8, 
                 drop_rate=0.2, 
                 drop_path_rate=0.2, 
                 attn_drop_rate=0.,
                 pretrained=os.path.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
                 use_head=False,
        ) :
        super().__init__()
        self.seqlen = seqlen
        self.input_proj = nn.Linear(2048, embed_dim)
        self.transformer = Transformer(depth=depth, embed_dim=embed_dim, mlp_hidden_dim=embed_dim*4.,
            h=h, drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate, length=seqlen)
        self.out_proj = nn.Linear(embed_dim, 2048)

        self.regressor = Regressor()
        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')

    def forward(self, img_feat, is_train=False, J_regressor=None, n_iter=3):
        """
        Return : [B, T, *]
        """
        B, T = img_feat.shape[:2]

        feature = self.input_proj(img_feat)
        feature = self.transformer(feature)             # [B, T, dim]
        feature_out = self.out_proj(feature)                # [B, T, 2048]

        smpl_output = self.regressor(feature_out, is_train=is_train, J_regressor=J_regressor, n_iter=n_iter)

        if is_train:
            size = T
        else :
            size = 1

        for s in smpl_output:
            s['theta'] = s['theta'].reshape(B, size, -1)
            s['verts'] = s['verts'].reshape(B, size, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(B, size, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(B, size, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(B, size, -1, 3, 3)

        return smpl_output, feature

def get_model(depth=3, length=16, embed_dim=512, mlp_hidden_dim=1024, h=8, drop_rate=0.2, drop_path_rate=0.2, attn_drop_rate=0.):
    model = THMR(depth=depth, seqlen=length, embed_dim=embed_dim)
    return model
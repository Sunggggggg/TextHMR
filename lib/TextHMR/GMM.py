import numpy as np
import os
import os.path as osp
import torch
import torch.nn as nn
from functools import partial

from lib.core.config import BASE_DATA_DIR
from lib.models.smpl import SMPL_MEAN_PARAMS
from lib.models.spin import Regressor
from lib.TextHMR.transformer import Block

class Transformer(nn.Module):
    def __init__(self, depth=3, embed_dim=512, mlp_hidden_dim=1024, 
                 h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=16):
        super().__init__()
        qkv_bias = True
        qk_scale = None
        # load mean smpl pose, shape and cam
        mean_params = np.load(SMPL_MEAN_PARAMS)
        self.init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0).to('cuda')
        self.init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0).to('cuda')
        self.init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0).to('cuda')
        
        self.mask_token_mlp = nn.Sequential(
            nn.Linear(24 * 6 + 13, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim // 2)
        )
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)]) 
        self.norm = norm_layer(embed_dim)

        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=embed_dim // 2, num_heads=h, mlp_hidden_dim=embed_dim * 2, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth // 2)])
        self.decoder_embed = nn.Linear(embed_dim, embed_dim // 2, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim // 2))
        self.decoder_norm = norm_layer(embed_dim // 2)


    def forward(self, x, is_train=True, mask_ratio=0.):
        if is_train:
            latent, mask, ids_restore = self.forward_encoder(x, mask_flag=True, mask_ratio=mask_ratio)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        else:
            latent, mask, ids_restore = self.forward_encoder(x, mask_flag=False,mask_ratio=0.)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        return pred, mask

    def forward_encoder(self, x, mask_flag=False, mask_ratio=0.):
        x = x + self.pos_embed
        if mask_flag:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
            # print('mask')
        else:
            mask = None
            ids_restore = None

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        if ids_restore is not None:
            mean_pose = torch.cat((self.init_pose, self.init_shape, self.init_cam), dim=-1)
            # append mask tokens to sequence
            mask_tokens = self.mask_token_mlp(mean_pose)
            mask_tokens = mask_tokens.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        else:
            x_ = x
        x = x_ + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device, dtype=torch.bool)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore).unsqueeze(-1) # assgin value from ids_restore
        
        return x_masked, mask, ids_restore

class GMM(nn.Module):
    def __init__(
            self,
            seqlen,
            n_layers=1,
            d_model=2048,
            num_head=8, 
            dropout=0., 
            drop_path_r=0.,
            atten_drop=0.,
            mask_ratio=0.,
            pretrained=osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
    ):

        super(GMM, self).__init__()
            
        self.proj = nn.Linear(2048, d_model)
        self.trans = Transformer(depth=n_layers, embed_dim=d_model, \
                mlp_hidden_dim=d_model*4, h=num_head, drop_rate=dropout, \
                drop_path_rate=drop_path_r, attn_drop_rate=atten_drop, length=seqlen)
        self.out_proj = nn.Linear(d_model // 2, 2048)
        self.mask_ratio = mask_ratio
        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor()
        self.initialize_weights()
        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')

    def forward(self, input, is_train=False, J_regressor=None):
        batch_size, seqlen = input.shape[:2]

        input = self.proj(input)
        if is_train:
            mem, mask_ids, ids_restore = self.trans.forward_encoder(input, mask_flag=True, mask_ratio=self.mask_ratio)
        else:
            mem, mask_ids, ids_restore = self.trans.forward_encoder(input, mask_flag=False, mask_ratio=0.)
        pred = self.trans.forward_decoder(mem, ids_restore)  # [N, L, p*p*3]

        if is_train:
            feature = self.out_proj(pred)
        else:
            feature = self.out_proj(pred)[:, seqlen // 2][:, None, :]

        smpl_output_global, pred_global = self.regressor(feature, is_train=is_train, J_regressor=J_regressor, n_iter=3)
        
        scores = None
        if is_train:
            size = seqlen
        else:
            size = 1

        for s in smpl_output_global:
            s['theta'] = s['theta'].reshape(batch_size, size, -1)
            s['verts'] = s['verts'].reshape(batch_size, size, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, size, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, size, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(batch_size, size, -1, 3, 3)
            s['scores'] = scores
       
        return smpl_output_global, mask_ids, mem, pred_global

    def initialize_weights(self):
        torch.nn.init.normal_(self.trans.pos_embed, std=.02)
        torch.nn.init.normal_(self.trans.decoder_pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

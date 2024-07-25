import torch
import torch.nn as nn
from .transformer import Attention, Mlp
from einops import rearrange
from timm.models.layers import DropPath

coco_head_idx = [0, 1, 2, 3, 4]
coco_left_arm_idx = [5, 7, 9]
coco_right_arm_idx = [6, 8, 10]
coco_left_leg_idx = [11, 13, 15]
coco_right_leg_idx = [12, 14, 16]

class PartAttentionModule(nn.Module):
    def __init__(self, 
                 data_type='coco',
                 embed_dim=256):
        super().__init__()
        # Part
        self.head_idx = eval(f'{data_type}_head_idx')
        self.left_arm_idx = eval(f'{data_type}_left_arm_idx')
        self.right_arm_idx = eval(f'{data_type}_right_arm_idx')
        self.left_leg_idx = eval(f'{data_type}_left_leg_idx')
        self.right_leg_idx = eval(f'{data_type}_right_leg_idx')

        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 5), padding=(1, 2)),
            nn.ReLU()
            )
        self.left_arm_conv = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU()
            )
        self.right_arm_conv = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU()
            )
        self.left_leg_conv = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU()
            )
        self.right_leg_conv = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU()
            )

    def forward(self, full_body):
        """
        full_body
            [B, T, J, dim]

        return 
            [B, T, J, dim]
        """
        B, T = full_body.shape[:2]
        head = full_body[:, :, self.head_idx]            # [B, T, 5, 256]
        left_arm = full_body[:, :, self.left_arm_idx]    # [B, T, 3, 256]
        right_arm = full_body[:, :, self.right_arm_idx]  # [B, T, 3, 256]
        left_leg = full_body[:, :, self.left_leg_idx]    # [B, T, 3, 256]
        right_leg = full_body[:, :, self.right_leg_idx]  # [B, T, 3, 256]

        # head
        enc_head = self.head_conv(head.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)                 # [B, T, 5, 256]
        enc_left_arm = self.left_arm_conv(left_arm.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)     # [B, T, 3, 256]
        enc_right_arm = self.right_arm_conv(right_arm.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)  # [B, T, 3, 256]
        enc_left_leg = self.left_leg_conv(left_leg.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)     # [B, T, 3, 256]
        enc_right_leg = self.right_leg_conv(right_leg.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)  # [B, T, 3, 256]

        # Skip
        full_body[:, :, self.head_idx] = enc_head
        full_body[:, :, self.left_arm_idx] = enc_left_arm
        full_body[:, :, self.right_arm_idx] = enc_right_arm
        full_body[:, :, self.left_leg_idx] = enc_left_leg
        full_body[:, :, self.right_leg_idx] = enc_right_leg
        
        return full_body

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads=8, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)
        self.part_atten = PartAttentionModule(embed_dim=embed_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.t_atten = Attention(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.s_atten = Attention(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=embed_dim*2., drop=drop)
    
    def forward(self, x):
        """
        full_body(x) : [B, T, J, dim]
        """
        B, T, J = x.shape[:3]
        x = x + self.drop_path(self.part_atten(self.norm1(x)))

        x = rearrange(x, 'B T J C -> (B J) T C')
        x = x + self.drop_path(self.t_atten(self.norm2(x)))
        x = rearrange(x, '(B J) T C -> (B T) J C', T=T, J=J)
        x = x + self.drop_path(self.s_atten(self.norm3(x)))

        x = x + self.drop_path(self.mlp(self.norm4(x)))
        
        x = rearrange(x, '(B T) J C -> B T J C', T=T)
        return x


class PartAttention_v2(nn.Module):
    def __init__(self, depth=3, embed_dim=512, mlp_hidden_dim=1024, 
                 h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., num_joints=17, num_frames=16) :
        super().__init__()
        self.num_joints = num_joints
        self.num_frames = num_frames
        qkv_bias = True
        qk_scale = None

        # 
        self.embeding = nn.Linear(2, embed_dim)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  

        self.blocks = nn.ModuleList([
            Block(embed_dim=embed_dim, num_heads=h, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=nn.LayerNorm)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        #    
    def forward(self, x):
        x = self.embeding(x)
        
        x = x + self.temporal_pos_embed[:, :, None].tile(1, 1, self.num_joints, 1) \
            + self.spatial_pos_embed[:, None, :].tile(1, self.num_frames, 1, 1)
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

class PartAttention(nn.Module):
    def __init__(self, data_type='coco', depth=4, embed_dim=256, num_joints=17, num_frames=16) :
        super().__init__()
        self.depth = depth
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.proj_input = nn.Linear(2, embed_dim)

        # Part
        self.head_idx = eval(f'{data_type}_head_idx')
        self.left_arm_idx = eval(f'{data_type}_left_arm_idx')
        self.right_arm_idx = eval(f'{data_type}_right_arm_idx')
        self.left_leg_idx = eval(f'{data_type}_left_leg_idx')
        self.right_leg_idx = eval(f'{data_type}_right_leg_idx')

        # Encoder 
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.pos_drop = nn.Dropout()

        self.part_atten = nn.ModuleList(
            PartAttentionModule(embed_dim=embed_dim)
            for _ in range(depth)
        )
        self.spatial_atten = nn.ModuleList(
            Attention(embed_dim, num_heads=8, attn_drop=0.2, proj_drop=0.1)
            for _ in range(depth)
        )

        self.temporal_atten = nn.ModuleList(
            Attention(embed_dim, num_heads=8, attn_drop=0.2, proj_drop=0.1)
            for _ in range(depth)
        )

        self.norm_s = nn.LayerNorm(embed_dim)
        self.norm_t = nn.LayerNorm(embed_dim)

        # Joint head
        self.head_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 3))
        self.head_left_arm = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 3))
        self.head_right_arm = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 3))
        self.head_left_leg = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 3))
        self.head_right_leg = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 3))
    
    def output_head(self, refine_body, body_feature):
        refine_body[:, :, self.head_idx] = self.head_head(body_feature[:, :, self.head_idx])
        refine_body[:, :, self.left_arm_idx] = self.head_left_arm(body_feature[:, :, self.left_arm_idx])
        refine_body[:, :, self.right_arm_idx] = self.head_right_arm(body_feature[:, :, self.right_arm_idx])
        refine_body[:, :, self.left_leg_idx] = self.head_left_leg(body_feature[:, :, self.left_leg_idx])
        refine_body[:, :, self.right_leg_idx] = self.head_right_leg(body_feature[:, :, self.right_leg_idx])

        return refine_body

    def forward(self, pose_2d, return_joint=False):
        """
        pose_2d : [B, T, J, 2]
        """
        global_body = local_body = self.proj_input(pose_2d)        # [B, T, J, 256]
        B, T, J = pose_2d.shape[:3]
        for idx, (part_attn, s_attn, t_attn) in enumerate(zip(self.part_atten, self.spatial_atten, self.temporal_atten)):
            if idx == 0 :
                global_body = rearrange(global_body, 'b t j c -> (b t) j c')
                global_body = global_body + self.spatial_pos_embed
                global_body = self.pos_drop(global_body)
                global_body = s_attn(global_body)
                global_body = self.norm_s(global_body)

                global_body = rearrange(global_body, '(b t) j c -> (b j) t c', t=self.num_frames)
                global_body = global_body + self.temporal_pos_embed
                global_body = self.pos_drop(global_body)
                global_body = t_attn(global_body)
                global_body = self.norm_t(global_body)
            else :
                global_body = rearrange(global_body, 'b t j c -> (b t) j c')
                global_body = s_attn(global_body)
                global_body = self.norm_s(global_body)

                global_body = rearrange(global_body, '(b t) j c -> (b j) t c', t=self.num_frames)
                global_body = t_attn(global_body)
                global_body = self.norm_t(global_body)
            
            global_body = rearrange(global_body, '(b j) t c -> b t j c', j=self.num_joints)
            local_body = part_attn(local_body)
            global_body = global_body + local_body
    
        if return_joint :
            refine_body = torch.zeros((B, T, J, 3), device=pose_2d.device)
            refine_body = self.output_head(refine_body, global_body)
            return refine_body
        else :
            return global_body
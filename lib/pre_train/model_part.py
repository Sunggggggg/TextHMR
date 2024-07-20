import torch
import torch.nn as nn
#from transformer import CrossAttention, Transformer

coco_head_idx = [0, 1, 2, 3, 4]
coco_left_arm_idx = [5, 7, 9]
coco_right_arm_idx = [6, 8, 10]
coco_left_leg_idx = [11, 13, 15]
coco_right_leg_idx = [12, 14, 16]

class PartAttention(nn.Module):
    def __init__(self, 
                 data_type='coco', 
                 seqlen=16,
                 embed_dim=256):
        super().__init__()

        # Part
        self.head_idx = eval(f'{data_type}_head_idx')
        self.left_arm_idx = eval(f'{data_type}_left_arm_idx')
        self.right_arm_idx = eval(f'{data_type}_right_arm_idx')
        self.left_leg_idx = eval(f'{data_type}_left_leg_idx')
        self.right_leg_idx = eval(f'{data_type}_right_leg_idx')

        self.head_conv = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim//2,
                                  kernel_size=(3, 5), padding=(1, 2))
        self.left_arm_conv = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim//2,
                                  kernel_size=(3, 3), padding=(1, 1))
        self.right_arm_conv = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim//2,
                                  kernel_size=(3, 3), padding=(1, 1))
        self.left_leg_conv = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim//2,
                                  kernel_size=(3, 3), padding=(1, 1))
        self.right_leg_conv = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim//2,
                                  kernel_size=(3, 3), padding=(1, 1))
        
        self.proj_dec = nn.Linear(embed_dim, embed_dim//2)
        self.left_arm_dec = nn.Conv1d(in_channels=6, out_channels=6//2, kernel_size=1)
        self.right_arm_dec = nn.Conv1d(in_channels=6, out_channels=6//2, kernel_size=1)
        self.left_leg_dec = nn.Conv1d(in_channels=6, out_channels=6//2, kernel_size=1)
        self.right_leg_dec = nn.Conv1d(in_channels=6, out_channels=6//2, kernel_size=1)

    def forward(self, pose):
        """
        pose : [B, T, J, c]
        """
        B, T = pose.shape[:2]
        full_body = pose
        head = pose[:, :, self.head_idx]            # [B, T, 5, c]
        left_arm = pose[:, :, self.left_arm_idx]    # [B, T, 3, c]
        right_arm = pose[:, :, self.right_arm_idx]  # [B, T, 3, c]
        left_leg = pose[:, :, self.left_leg_idx]    # [B, T, 3, c]
        right_leg = pose[:, :, self.right_leg_idx]  # [B, T, 3, c]

        # head
        enc_left_arm = self.head_conv(head.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)     # [B, T, 5, c]

        # left_arm
        enc_left_arm = self.left_arm_conv(left_arm.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)     # [B, T, 3, c]

        # right_arm
        enc_right_arm = self.right_arm_conv(right_arm.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)  # [B, T, 3, c]

        # left_leg
        enc_left_leg = self.left_leg_conv(left_leg.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)     # [B, T, 3, c]

        # right_leg
        enc_right_leg = self.right_leg_conv(right_leg.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)  # [B, T, 3, c]

        # Concat 
        full_body = self.proj_dec(full_body)
        dec_head = full_body[:, :, self.head_idx]            # [B, T, 5, c]
        dec_left_arm = full_body[:, :, self.left_arm_idx]    # [B, T, 3, c]
        dec_right_arm = full_body[:, :, self.right_arm_idx]  # [B, T, 3, c]
        dec_left_leg = full_body[:, :, self.left_leg_idx]    # [B, T, 3, c]
        dec_right_leg = full_body[:, :, self.right_leg_idx]  # [B, T, 3, c]

        enc_left_arm = torch.cat([full_body[:, :, self.left_arm_idx], enc_left_arm], dim=2)     # [B, T, 3+3, c]
        enc_right_arm = torch.cat([full_body[:, :, self.right_arm_idx], enc_right_arm], dim=2)  # [B, T, 3+3, c]
        enc_left_leg = torch.cat([full_body[:, :, self.left_leg_idx], enc_left_leg], dim=2)     # [B, T, 3+3, c]
        enc_right_leg = torch.cat([full_body[:, :, self.right_leg_idx], enc_right_leg], dim=2)  # [B, T, 3+3, c]
        
        full_body[:, :, self.left_arm_idx] = left_arm + self.left_arm_dec(enc_left_arm.reshape(B*T, 6, -1)).reshape(B, T, 3, -1)         # [BT, 3, c]
        full_body[:, :, self.right_arm_idx] = right_arm + self.right_arm_dec(enc_right_arm.reshape(B*T, 6, -1)).reshape(B, T, 3, -1)     # [BT, 3, c]
        full_body[:, :, self.left_leg_idx] = left_leg + self.left_leg_dec(enc_left_leg.reshape(B*T, 6, -1)).reshape(B, T, 3, -1)         # [BT, 3, c]
        full_body[:, :, self.right_leg_idx] = right_leg + self.right_leg_dec(enc_right_leg.reshape(B*T, 6, -1)).reshape(B, T, 3, -1)     # [BT, 3, c]
        
        print(full_body.shape)
        return
    
pose2d = torch.rand((4, 16, 17, 256))
model = PartAttention('coco')

model(pose2d)

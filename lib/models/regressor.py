import torch
import torch.nn as nn

from .lifter import add_joint
from lib.data_utils._kp_utils import convert_kps

def extract_coco_kp3d(init_theta, joint_guide):
    """
    init_theta  : {}
    joint_guide : [B, J, 3] J=19
    """
    init_theta = init_theta[-1]
    kp_3d = init_theta['kp_3d'] # [BT, 49, 3] T=1
    coco_kp3d = add_joint(convert_kps(kp_3d, src='spin', dst='coco'))  # [B, 19, 3]
    return coco_kp3d


class Semantic_Regrssor(nn.Module):
    def __init__(self) :
        super().__init__()
        self.head = nn.Conv1d(in_channels=36, out_channels=1, kernel_size=1)

        

    
    def forward(self, init_theta, semantic_guide):
        """
        semantic_guide : [B, N, dim] N=36
        """
        semantic_guide = self.head(semantic_guide)  # [B, 1, dim]


        return 
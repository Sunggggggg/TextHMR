import torch
from lib.SemHMR.model import Model

img_feat = torch.rand((1, 16, 2048))
pose3d = torch.rand((1, 19, 3))
text_embed = torch.rand((1, 36, 768))
idx = torch.tensor([[14]])
model = Model()
_ = model(text_embed, img_feat, pose3d, idx)
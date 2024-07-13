import torch
from lib.pre_train.model import Model

img_feat = torch.rand((1, 16, 2048))
pose3d = torch.rand((1, 16, 17, 2))
text_embed = torch.rand((1, 16, 768))
idx = torch.tensor([[14]])
model = Model(num_total_motion=7000)
pred_text, pred_kp_3d = model(pose3d, text_embed)

print(pred_kp_3d.shape, pred_text.shape)
net_params = sum(map(lambda x: x.numel(), model.parameters()))
print(net_params)
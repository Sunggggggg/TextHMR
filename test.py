import torch
import lib.models.linker as linker

pose3d = torch.rand((1, 19, 3))
text_embed = torch.rand((1, 36, 256))
model = linker.get_model()
x, y = model(pose3d, text_embed)

print(x.shape, y.shape)
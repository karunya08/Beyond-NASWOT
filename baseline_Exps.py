from models import get_cell_based_tiny_net
import torch

cfg = {
    'name': 'infer.tiny',
    'C': 16,
    'N': 5,
    'num_classes': 1,
    'arch_str': '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|'
}

net = get_cell_based_tiny_net(cfg)

print(net(torch.randn(2, 3, 32, 32))[0].shape)
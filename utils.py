import numpy as np
import torch
import torch.nn.functional as F

def get_activation_mask(inp, mask_type='relu'):
    """Return an activation derivative mask for the given input tensor.

    Args:
        inp: flattened activation tensor (batch_size, features).
        mask_type: one of 'relu', 'swish', 'hardmish'.

    Returns:
        Tensor of the same shape as inp containing the derivative values.
    """
    if mask_type == 'relu':
        # ReLU derivative: 1 where x > 0, else 0 (original NASWOT)
        return (inp > 0).float()
    elif mask_type == 'swish':
        # Swish / SiLU: f(x) = x * sigmoid(x)
        # f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        sig = torch.sigmoid(inp)
        return sig + inp * sig * (1.0 - sig)
    elif mask_type == 'hardmish':
        # Hard Mish: f(x) = 0 if x<=-2, x*(x+2)/2 if -2<x<0, x if x>=0
        # f'(x) = 0 if x<=-2, x+1 if -2<x<0, 1 if x>=0
        return torch.where(
            inp >= 0,
            torch.ones_like(inp),
            torch.where(
                inp > -2,
                inp + 1.0,
                torch.zeros_like(inp),
            ),
        )
    else:
        raise ValueError(f"Unknown mask_type '{mask_type}'. Choose from: relu, swish, hardmish")


def get_binary_tau_mask(inp, threshold=0.0):
    """Return a binary mask by thresholding activations (original NASWOT method).

    Args:
        inp: flattened activation tensor (batch_size, features).
        threshold: cutoff value; activations above this become 1, else 0.

    Returns:
        Binary tensor of the same shape as inp.
    """
    return (inp > threshold).float()


class DropChannel(torch.nn.Module):
    def __init__(self, p, mod):
        super(DropChannel, self).__init__()
        self.mod = mod
        self.p = p
    def forward(self, s0, s1, droppath):
        ret = self.mod(s0, s1, droppath)
        return ret


class DropConnect(torch.nn.Module):
    def __init__(self, p):
        super(DropConnect, self).__init__()
        self.p = p
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        dim1 = inputs.shape[2]
        dim2 = inputs.shape[3]
        channel_size = inputs.shape[1]
        keep_prob = 1 - self.p
        # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
        random_tensor = keep_prob
        random_tensor += torch.rand([batch_size, channel_size, 1, 1], dtype=inputs.dtype, device=inputs.device)
        binary_tensor = torch.floor(random_tensor)
        output = inputs / keep_prob * binary_tensor
        return output    

def add_dropout(network, p, prefix=''):
    #p = 0.5
    for attr_str in dir(network):
        target_attr = getattr(network, attr_str)
        if isinstance(target_attr, torch.nn.Conv2d):
            setattr(network, attr_str, torch.nn.Sequential(target_attr, DropConnect(p)))
        elif isinstance(target_attr, Cell):
            setattr(network, attr_str, DropChannel(p, target_attr))
    for n, ch in list(network.named_children()):
        #print(f'{prefix}add_dropout {n}')
        if isinstance(ch, torch.nn.Conv2d):
            setattr(network, n, torch.nn.Sequential(ch, DropConnect(p)))
        elif isinstance(ch, Cell):
            setattr(network, n, DropChannel(p, ch))
        else:
            add_dropout(ch, p, prefix + '\t')
             



def orth_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.orthogonal_(m.weight)

def uni_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.uniform_(m.weight)

def uni2_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.uniform_(m.weight, -1., 1.)

def uni3_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.uniform_(m.weight, -.5, .5)

def norm_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.norm_(m.weight)

def eye_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.eye_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.dirac_(m.weight)



def fixup_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.zero_(m.weight)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.zero_(m.weight)
        torch.nn.init.zero_(m.bias)


def init_network(network, init):
    if init == 'orthogonal':
        network.apply(orth_init)
    elif init == 'uniform':
        print('uniform')
        network.apply(uni_init)
    elif init == 'uniform2':
        network.apply(uni2_init)
    elif init == 'uniform3':
        network.apply(uni3_init)
    elif init == 'normal':
        network.apply(norm_init)
    elif init == 'identity':
        network.apply(eye_init)

import argparse
import nasspace
import datasets
import random
import numpy as np
import os
import torch
from scores import get_score_func
from scipy import stats
from utils import (add_dropout, init_network, get_activation_mask,
                   get_binary_tau_mask, replace_relu, ACTIVATION_MODULES)

parser = argparse.ArgumentParser(description='NAS Without Training')
parser.add_argument('--data_loc', default='../cifardata/', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='../NAS-Bench-201-v1_0-e61699.pth',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results', type=str, help='folder to save results')
parser.add_argument('--save_string', default='naswot', type=str, help='prefix of results file')
parser.add_argument('--score', default='hook_logdet', type=str, help='the score to evaluate')
parser.add_argument('--nasspace', default='nasbench201', type=str, help='the nas search space to use')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
parser.add_argument('--sigma', default=0.05, type=float, help='noise level if augtype is "gaussnoise"')
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--init', default='', type=str)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--maxofn', default=1, type=int, help='score is the max of this many evaluations of the network')
parser.add_argument('--n_samples', default=100, type=int)
parser.add_argument('--n_runs', default=500, type=int)
parser.add_argument('--stem_out_channels', default=16, type=int, help='output channels of stem convolution (nasbench101)')
parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules (nasbench101)')
parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack (nasbench101)')
parser.add_argument('--num_labels', default=1, type=int, help='#classes (nasbench101)')
parser.add_argument('--method', default='binary_tau', type=str, choices=['binary_tau', 'masks'],
                    help='scoring method: binary_tau (original NASWOT) or masks (derivative masks)')
parser.add_argument('--mask_type', default='relu', type=str, choices=['relu', 'hardswish', 'mish'],
                    help='activation derivative mask, used when --method=masks (default: relu)')
parser.add_argument('--threshold', default=0.0, type=float,
                    help='threshold for binary_tau method (default: 0.0)')
parser.add_argument('--max_archs', default=0, type=int,
                    help='max architectures to score (0 = all)')
parser.add_argument('--ground_truth', default='', type=str,
                    help='path prefix to ground truth .npy files from train_ground_truth.py '
                         '(e.g. ground_truth/silu_100archs_200ep). When set, only scores the '
                         'architectures in _uids.npy and uses _accs.npy for Kendall-tau.')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def get_batch_jacobian(net, x, target, device, args=None):
    net.zero_grad()
    x.requires_grad_(True)
    y, out = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach(), y.detach(), out.detach()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print(f'Using GPU {args.GPU}: {torch.cuda.get_device_name(0)} (PyTorch CUDA {torch.version.cuda})', flush=True)
else:
    print(f'CUDA unavailable; using CPU (PyTorch {torch.__version__}, CUDA build: {torch.version.cuda})', flush=True)
searchspace = nasspace.get_search_space(args)
savedataset = args.dataset
dataset = 'fake' if 'fake' in args.dataset else args.dataset
args.dataset = args.dataset.replace('fake', '')
if args.dataset == 'cifar10':
    args.dataset = args.dataset + '-valid'
if 'valid' in args.dataset:
    args.dataset = args.dataset.replace('-valid', '')
train_loader = datasets.get_data(args.dataset, args.data_loc, args.trainval, args.batch_size, args.augtype, args.repeat, args)
os.makedirs(args.save_loc, exist_ok=True)

filename = f'{args.save_loc}/{args.save_string}_{args.score}_{args.nasspace}_{savedataset}{"_" + args.init + "_" if args.init != "" else args.init}_{"_dropout" if args.dropout else ""}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.maxofn}_{args.seed}'
accfilename = f'{args.save_loc}/{args.save_string}_accs_{args.nasspace}_{savedataset}_{args.trainval}'

if args.dataset == 'cifar10':
    acc_type = 'ori-test'
    val_acc_type = 'x-valid'
else:
    acc_type = 'x-test'
    val_acc_type = 'x-valid'

# ---- Ground truth mode vs full-bench mode ----
if args.ground_truth:
    gt_uids = np.load(args.ground_truth + '_uids.npy')
    gt_accs = np.load(args.ground_truth + '_accs.npy')
    # Filter out architectures that haven't been trained yet (acc == 0)
    trained_mask = gt_accs > 0
    gt_uids = gt_uids[trained_mask]
    gt_accs = gt_accs[trained_mask]
    n_archs = len(gt_uids)
    print(f'Ground truth mode: scoring {n_archs} architectures from {args.ground_truth}')
    scores = np.zeros(n_archs)
    accs = gt_accs
else:
    gt_uids = None
    n_archs = len(searchspace)
    scores = np.zeros(n_archs)
    try:
        accs = np.load(accfilename + '.npy')
    except:
        accs = np.zeros(n_archs)

# Determine which activation module to hook based on method/mask_type
if args.method == 'masks' and args.mask_type != 'relu':
    hook_act_class = ACTIVATION_MODULES[args.mask_type]
    hook_act_name = type(hook_act_class()).__name__   # e.g. 'SiLU', 'HardMish'
    swap_activation = True
    print(f'Activation swap: ReLU -> {hook_act_name} (mask_type={args.mask_type})')
else:
    hook_act_name = 'ReLU'
    swap_activation = False

# ---- Main scoring loop ----
def iterate_architectures():
    """Yield (loop_index, uid, network) for each architecture to score."""
    if gt_uids is not None:
        for i, uid in enumerate(gt_uids):
            if args.max_archs > 0 and i >= args.max_archs:
                return
            network = searchspace.get_network(int(uid))
            yield i, int(uid), network
    else:
        for i, (uid, network) in enumerate(searchspace):
            if args.max_archs > 0 and i >= args.max_archs:
                return
            yield i, uid, network


for i, uid, network in iterate_architectures():
    try:
        if args.dropout:
            add_dropout(network, args.sigma)
        if args.init != '':
            init_network(network, args.init)

        # Swap activations for non-ReLU scoring
        if swap_activation:
            replace_relu(network, ACTIVATION_MODULES[args.mask_type])

        if 'hook_' in args.score:
            network.K = np.zeros((args.batch_size, args.batch_size))
            def counting_forward_hook(module, inp, out):
                try:
                    if not module.visited_backwards:
                        return
                    if isinstance(inp, tuple):
                        inp = inp[0]
                    inp = inp.view(inp.size(0), -1)
                    if args.method == 'binary_tau':
                        x = get_binary_tau_mask(inp, args.threshold)
                    else:
                        x = get_activation_mask(inp, args.mask_type)
                    K = x @ x.t()
                    K2 = (1.-x) @ (1.-x.t())
                    network.K = network.K + K.detach().cpu().numpy() + K2.detach().cpu().numpy()
                except Exception as e:
                    # print(f"  [HOOK ERROR] {type(e).__name__}: {e}")
                    pass

                
            def counting_backward_hook(module, grad_input, grad_output):
                # Updated signature for register_full_backward_hook (PyTorch 2.x)
                module.visited_backwards = True

            for m in network.modules():
                if hasattr(m, "inplace"):
                    m.inplace = False    
            for name, module in network.named_modules():
                if hook_act_name in str(type(module)):
                    module.register_forward_hook(counting_forward_hook)
                    # Use register_full_backward_hook for PyTorch 2.x compatibility
                    module.register_full_backward_hook(counting_backward_hook)

        network = network.to(device)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        s = []
        for j in range(args.maxofn):
            data_iterator = iter(train_loader)
            x, target = next(data_iterator)
            x2 = torch.clone(x)
            x2 = x2.to(device)
            x, target = x.to(device), target.to(device)
            jacobs, labels, y, out = get_batch_jacobian(network, x, target, device, args)





            if 'hook_' in args.score:
                network(x2.to(device))
                s.append(get_score_func(args.score)(network.K, target))
            else:
                s.append(get_score_func(args.score)(jacobs, labels))
        scores[i] = np.mean(s)
        if gt_uids is None:
            # Full-bench mode: get accuracy from NAS-Bench-201 API
            accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)
        # Ground truth mode: accs already loaded from .npy
        accs_ = accs[~np.isnan(scores)]
        scores_ = scores[~np.isnan(scores)]
        numnan = np.isnan(scores).sum()
        tau, p = stats.kendalltau(accs_[:max(i-numnan, 1)], scores_[:max(i-numnan, 1)])
        print(f'[{i+1}/{n_archs}] uid={uid:5d} | score={scores[i]:.4f} | tau={tau:.4f}')
        if i % 1000 == 0:
            np.save(filename, scores)
            np.save(accfilename, accs)
    except Exception as e:
        print(e)
        if gt_uids is None:
            accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)
        scores[i] = np.nan
np.save(filename, scores)
np.save(accfilename, accs)

# Final Kendall-tau -- only count architectures that were actually scored
n_scored = sum(1 for idx in range(n_archs)
               if not np.isnan(scores[idx]) and (idx < (args.max_archs if args.max_archs > 0 else n_archs)))
scored_mask = np.zeros(n_archs, dtype=bool)
scored_mask[:n_scored] = True
scored_mask &= ~np.isnan(scores)

if scored_mask.sum() > 1:
    final_tau, final_p = stats.kendalltau(accs[scored_mask], scores[scored_mask])
    print(f'\nFinal Kendall-tau: {final_tau:.4f} (p={final_p:.2e}, n={scored_mask.sum()})')

from models import get_cell_based_tiny_net, get_search_spaces
from nas_201_api import NASBench201API as API
import itertools
import random
import numpy as np
from models.cell_searchs.genotypes import Structure
from copy import deepcopy
import json
import torch


class Nasbench201:
    def __init__(self, dataset, apiloc):
        self.dataset = dataset
        self.api = API(apiloc, verbose=False)
        self.epochs = '12'
    def get_network(self, uid):
        config = self.api.get_net_config(uid, 'cifar10-valid')
        config['num_classes'] = 1
        network = get_cell_based_tiny_net(config)
        return network
    def __iter__(self):
        for uid in range(len(self)):
            network = self.get_network(uid)
            yield uid, network
    def __getitem__(self, index):
        return index
    def __len__(self):
        return 15625
    def num_activations(self):
        network = self.get_network(0)
        return network.classifier.in_features

    def get_12epoch_accuracy(self, uid, acc_type, trainval, traincifar10=False):
        info = self.api.get_more_info(uid, 'cifar10-valid', iepoch=None, hp=self.epochs, is_random=True)
        return info['valid-accuracy']
    
    def get_final_accuracy(self, uid, acc_type, trainval):
        if self.dataset == 'cifar10' and trainval:
            info = self.api.query_meta_info_by_index(uid, hp='200').get_metrics('cifar10-valid', 'x-valid')
        else:
            info = self.api.query_meta_info_by_index(uid, hp='200').get_metrics(self.dataset, acc_type)
        return info['accuracy']

    def get_accuracy(self, uid, acc_type, trainval=True):
        archinfo = self.api.query_meta_info_by_index(uid)
        if self.dataset == 'cifar10' and trainval:
            return archinfo.get_metrics('cifar10-valid', acc_type)['accuracy']
        else:
            return archinfo.get_metrics(self.dataset, acc_type)['accuracy']

    def get_accuracy_for_all_datasets(self, uid):
        archinfo = self.api.query_meta_info_by_index(uid,hp='200')

        c10 = archinfo.get_metrics('cifar10', 'ori-test')['accuracy']
        c10_val = archinfo.get_metrics('cifar10-valid', 'x-valid')['accuracy']

        c100 = archinfo.get_metrics('cifar100', 'x-test')['accuracy']
        c100_val = archinfo.get_metrics('cifar100', 'x-valid')['accuracy']

        imagenet = archinfo.get_metrics('ImageNet16-120', 'x-test')['accuracy']
        imagenet_val = archinfo.get_metrics('ImageNet16-120', 'x-valid')['accuracy']

        return c10, c10_val, c100, c100_val, imagenet, imagenet_val

    def train_and_eval(self, arch, dataname, acc_type, trainval=True, traincifar10=False):
        unique_hash = self.__getitem__(arch)
        time = self.get_training_time(unique_hash)
        acc12 = self.get_12epoch_accuracy(unique_hash, acc_type, trainval, traincifar10)
        acc = self.get_final_accuracy(unique_hash, acc_type, trainval)
        return acc12, acc, time

    def random_arch(self):
        return random.randint(0, len(self)-1)
    
    def get_training_time(self, unique_hash):
        info = self.api.get_more_info(unique_hash, 'cifar10-valid', iepoch=None, hp='12', is_random=True)
        return info['train-all-time'] + info['valid-per-time']
    
    def mutate_arch(self, arch):
        op_names = get_search_spaces('cell', 'nas-bench-201')
        config = self.api.get_net_config(arch, 'cifar10-valid')
        parent_arch = Structure(self.api.str2lists(config['arch_str']))
        child_arch = deepcopy( parent_arch )
        node_id = random.randint(0, len(child_arch.nodes)-1)
        node_info = list( child_arch.nodes[node_id] )
        snode_id = random.randint(0, len(node_info)-1)
        xop = random.choice( op_names )
        while xop == node_info[snode_id][0]:
          xop = random.choice( op_names )
        node_info[snode_id] = (xop, node_info[snode_id][1])
        child_arch.nodes[node_id] = tuple( node_info )
        arch_index = self.api.query_index_by_arch( child_arch )
        return arch_index



class ReturnFeatureLayer(torch.nn.Module):
    def __init__(self, mod):
        super(ReturnFeatureLayer, self).__init__()
        self.mod = mod
    def forward(self, x):
        return self.mod(x), x
                

def return_feature_layer(network, prefix=''):
    for n, ch in list(network.named_children()):
        if isinstance(ch, torch.nn.Linear):
            setattr(network, n, ReturnFeatureLayer(ch))
        else:
            return_feature_layer(ch, prefix + '\t')
             


def get_search_space(args):
    if args.nasspace == 'nasbench201':
        return Nasbench201(args.dataset, args.api_loc)
    


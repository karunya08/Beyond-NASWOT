##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
from config_utils import dict2config
from .cell_searchs import CellStructure

__all__ = ['get_cell_based_tiny_net', 'get_search_spaces', 'CellStructure']


def get_search_spaces(xtype, name):
  if xtype == 'cell':
    from .cell_operations import SearchSpaceNames
    assert name in SearchSpaceNames, 'invalid name [{:}] in {:}'.format(name, SearchSpaceNames.keys())
    return SearchSpaceNames[name]
  else:
    raise ValueError('invalid search-space type is {:}'.format(xtype))


# Build an NB201 network (TinyNetwork) from an architecture config
def get_cell_based_tiny_net(config):
  if isinstance(config, dict): config = dict2config(config, None)
  if config.name == 'infer.tiny':
    from .cell_infers import TinyNetwork
    if hasattr(config, 'genotype'):
      genotype = config.genotype
    elif hasattr(config, 'arch_str'):
      genotype = CellStructure.str2structure(config.arch_str)
    else:
      raise ValueError('Can not find genotype from this config : {:}'.format(config))
    return TinyNetwork(config.C, config.N, genotype, config.num_classes)
  else:
    raise ValueError('invalid network name : {:}'.format(config.name))
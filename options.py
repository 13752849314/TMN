import os

import yaml
from collections import OrderedDict

from utils.common import makedir

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def OrderedYaml():
    """
    yaml orderedDict support
    :return:
    """
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_represent(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_represent)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


loader, dumper = OrderedYaml()


def parse(opt_path):
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=loader)

    opt['result_path'] = os.path.join(opt['result_path'], opt['name'])
    opt['model_path'] = os.path.join(opt['model_path'], opt['name'])
    opt['logs_path'] = os.path.join(opt['logs_path'], opt['name'])
    makedir([opt['result_path'], opt['model_path'], opt['logs_path']])

    if opt['use_cuda']:
        gpu_list = ','.join((str(x) for x in opt['gpu_idx']))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    else:
        print('using cpu train!')

    print(f"{opt['name']} options loading done!")
    return opt


def dict2str(opt, indent_l=1):
    """
    dict to string for logger
    """
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


if __name__ == '__main__':
    opt1 = parse(r'./config/MTN_x4.yml')
    print(opt1)

import argparse
import yaml
import os
from easydict import EasyDict


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='path to config file')

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')

    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888)

    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set extra config keys')

    args = parser.parse_args()     # namespace
    args = EasyDict(args.__dict__)     # easydict

    assert args.config is not None
    with open(args.config, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)     # dict
        except:
            config = yaml.load(f)
        config = EasyDict(config)     # easydict

    for key in config:
        for k, v in config[key].items():
            args[k] = v

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, args)

    args.exp_path = os.path.join('exp', args.dataset, args.model_name, args.config.split('/')[-1][:-5])
    return args


def cfg_from_list(cfg_list, config):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = config
        for subkey in key_list[:-1]:
            assert subkey in d, f'NotFoundKey: {subkey}'
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, f'NotFoundKey: {subkey}'
        try:
            value = literal_eval(v)
        except:
            value = v

        if type(value) != type(d[subkey]) and isinstance(d[subkey], EasyDict):
            key_val_list = value.split(',')
            for src in key_val_list:
                cur_key, cur_val = src.split(':')
                val_type = type(d[subkey][cur_key])
                cur_val = val_type(cur_val)
                d[subkey][cur_key] = cur_val
        elif type(value) != type(d[subkey]) and isinstance(d[subkey], list):
            val_list = value.split(',')
            for k, x in enumerate(val_list):
                val_list[k] = type(d[subkey][0])(x)
            d[subkey] = val_list
        else:
            assert type(value) == type(d[subkey]), f'type {type(value)} does not match original type {type(d[subkey])}'
            d[subkey] = value

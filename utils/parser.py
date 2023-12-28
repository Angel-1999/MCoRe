import os
import yaml
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--archs', type = str, choices=['TSA'], help = 'our approach')
    parser.add_argument('--benchmark', type = str, choices=['FineDiving'], help = 'dataset')
    parser.add_argument('--prefix', type = str, default='default', help = 'experiment name')
    parser.add_argument('--resume', action='store_true', default=False ,help = 'resume training (interrupted by accident)')
    parser.add_argument('--sync_bn', type=bool, default=False)
    parser.add_argument('--fix_bn', type=bool, default=True)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--ckpts', type=str, default=None, help='test used ckpt path')
    parser.add_argument(
        '--feature_arch', type=str, choices=[
            # From torchvision
            'rn18',
            'rn18_tsm',
            'rn18_gsm',
            'rn50',
            'rn50_tsm',
            'rn50_gsm',

            # From timm (following its naming conventions)
            'rny002',
            'rny002_tsm',
            'rny002_gsm',
            'rny008',
            'rny008_tsm',
            'rny008_gsm',

            # From timm
            'convnextt',
            'convnextt_tsm',
            'convnextt_gsm'
        ], default='rny002_gsm', help='CNN architecture for feature extraction')
    parser.add_argument(
        '--temporal_arch', type=str, default='gru',
        choices=['', 'gru', 'deeper_gru', 'mstcn', 'asformer'],
        help='Spotting architecture, after spatial pooling')
    parser.add_argument('--clip_len', type=int, default=96)
    parser.add_argument('--modality', type=str, choices=['rgb', 'bw', 'flow'],
                        default='rgb')
    parser.add_argument('-mgpu', '--gpu_parallel', default=False, action='store_true')


    args = parser.parse_args()

    if args.test:
        if args.ckpts is None:
            raise RuntimeError('--ckpts should not be None when --test is activate')

    return args

def setup(args):

    args.config = '{}_TSA.yaml'.format(args.benchmark)
    args.experiment_path = os.path.join('./experiments', args.benchmark, args.prefix)
    if args.resume:

        cfg_path = os.path.join(args.experiment_path,'config.yaml')
        if not os.path.exists(cfg_path):
            print("Failed to resume")
            args.resume = False
            setup(args)
            return

        print('Resume yaml from %s' % cfg_path)
        with open(cfg_path) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        merge_config(config, args)
        args.resume = True
    else:
        config = get_config(args)
        merge_config(config, args)
        create_experiment_dir(args)
        save_experiment_config(args)

def get_config(args):
    try:
        print('Load config yaml from %s' % args.config)
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.Loader)
    except:
        raise NotImplementedError('%s arch is not supported')
    return config

def merge_config(config, args):
    for k, v in config.items():
        setattr(args, k, v)   

def create_experiment_dir(args):
    try:
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    except:
        pass
    
def save_experiment_config(args):
    config_path = os.path.join(args.experiment_path,'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(args.__dict__, f)
        print('Save the Config file at %s' % config_path)